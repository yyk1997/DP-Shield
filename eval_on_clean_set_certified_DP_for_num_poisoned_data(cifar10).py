'''codes used to evalate the certified robustness for the number of poisoned samples in training dataset (cifar10)
'''
import argparse
import math
import os, sys
import time
from tqdm import tqdm
from utils import default_args, imagenet, supervisor
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from scipy.stats import norm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import scipy.stats as stats
import torch
from lightly.models.modules import heads
from collections import OrderedDict
from scipy.special import comb
from diffusers import DDIMPipeline, DDPMPipeline



parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='none',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

# 数据集和后门攻击相关
args.dataset = "cifar10"
args.poison_type = "badnet"
dataset_size = 50000
args.poison_rate = 0.0002
# args.cover_rate = 0.002
# args.alpha = 0.2
# args.trigger = "badnet_patch4_dup_32.png"

# delta是触发器的有界范数
args.delta = 10000

# DP相关参数
MAX_GRAD_NORM = 1.0
Noise_Multi = 5.0
DELTA = 1e-5

# 可验证相关
args.N_m = 1000
args.bagging = 5000
args.eta = 0.001

# diffusion 参数
model_id = 'google/ddpm-cifar10-32'
ddpm = DDPMPipeline.from_pretrained(model_id).to("cuda")
args.noise_std = 0.5
test_noise_std = 1.0
test_folder_path = './poisoned_train_set/cifar10/denoised_test_set'

# args.inference_method = 'multinomial_label' or 'probability_scores'
args.inference_method = 'multinomial_label'

# args.certified_method = 'DP' or 'hypothesis'
args.certified_method = 'DP'

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools


if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default[args.poison_type]
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers[args.poison_type]



pretrain_name = os.path.join(supervisor.get_poison_set_dir(args), f'noise_std=0/pretrain_model_500_epochs.pth')
# pretrain_name = './poisoned_train_set/cifar10/none_0.000_10000_poison_seed=0/noise_std=0/pretrain_model_500_epochs.pth'


# diffusion model 计算
scheduler = ddpm.scheduler

# 获取 alpha_t 的累积乘积 (\(\bar{\alpha}_t\)) 序列
alphas_cumprod = scheduler.alphas_cumprod
# print("Alphas Cumulative Product Sequence:", alphas_cumprod)

# 如果需要每个时间步 t 的 alpha_t，可以计算
alphas = torch.sqrt(alphas_cumprod)  # 每步的 alpha_t
# print("Alphas Sequence:", alphas)

# 依照添加的噪声计算所预测的alpha_t_predict
alpha_t_predict = 1 / (args.noise_std **2 +1)

# 计算alpha_t_predict与序列中值的差异
differences = np.abs(alphas_cumprod - alpha_t_predict)

# 找到最小距离对应的位置和值
closest_index = np.argmin(differences)
closest_value = alphas_cumprod[closest_index]

closest_value = torch.sqrt(closest_value)




class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z




all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True

tools.setup_seed(args.seed)

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (
    supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

# 定义对于cifar10数据集的归一化操作
if args.dataset == 'cifar10':

    data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
# 定义对于gtsrb数据集的归一化操作
elif args.dataset == 'gtsrb':

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

elif args.dataset == 'imagenet':
    print('[ImageNet]')

elif args.dataset == 'ember':
    print('[Non-image Dataset] Ember')
else:
    raise NotImplementedError('dataset %s not supported' % args.dataset)

# 定义模型训练阶段各重要参数，包括epoch, learning_rate, batch_size等
if args.dataset == 'cifar10':

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 512

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 128

elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 90
    milestones = torch.tensor([30, 60])
    learning_rate = 0.1
    batch_size = 256

elif args.dataset == 'ember':

    num_classes = 2
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    milestones = torch.tensor([])
    batch_size = 512

else:

    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 0, 'pin_memory': True}

# 设置测试及评估数据
# 设置测试数据

if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')

    denoised_test_set_path = os.path.join(test_folder_path, f"denoised_test_set{test_noise_std}.pt")
    if os.path.exists(denoised_test_set_path):
        denoised_test_set = torch.load(denoised_test_set_path)
    else:
        # 干净测试数据中加入噪声
        test_set = tools.IMG_Dataset_Noise_and_Diffusion(data_dir=test_set_img_dir,
                                     label_path=test_set_label_path, alpha = closest_value,
                                         noise_std=test_noise_std)

        # 使用diffusion model对测试数据进行去噪
        denoised_test_set = tools.DenoisedPoisonedDataset(test_set, ddpm, closest_step = closest_index, scheduler=scheduler, transforms=data_transform)
        torch.save(denoised_test_set, denoised_test_set_path)




def certificate_over_dataset(model, num_classes, dataloader, args, N_m = args.N_m):
    model_preds = []
    labs = []
    model.cuda()
    softmax = nn.Softmax(dim= 1)
    with torch.no_grad():
        for _ in tqdm(range(N_m)):
            # 加载第_个模型
            # 加载保存的 state_dict
            checkpoint = torch.load(folder_path + f'/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/{_}.pt')

            # 检查是否有 "_module." 前缀
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k.startswith("_module."):
                    new_state_dict[k[len("_module."):]] = v  # 去掉 "_module." 前缀
                else:
                    new_state_dict[k] = v

            # 加载修正后的 state_dict 到模型
            model.projection_head.load_state_dict(new_state_dict)

            # model.load_state_dict(torch.load(folder_path + f'/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/{_}.pt'))
            all_pred = np.zeros((0, num_classes))

            # 遍历数据集
            for data, label in dataloader:

                if _ == 0:
                    labs = labs + list(label.numpy())

                # 获得模型的预测向量
                x_in = data.cuda()
                pred = model(x_in)
                pred = softmax(pred)
                # 将一个模型所有的预测结果concatenate到一起
                all_pred = np.concatenate([all_pred, pred.cpu().detach().numpy()], axis=0)

            # 保存每个模型对于所有样本的预测结果，存储到model_preds列表中
            model_preds.append(all_pred)

    # if args.inference_method == 'probability_scores':
    # 计算平均预测概率
    gx = np.array(model_preds).mean(0)

    labs = np.array(labs)

    # 计算每个样本中最高的预测概率
    pa_1 = gx.max(1)
    # 得到预测标签
    pred_c = gx.argmax(1)
    # 将最大预测概率设置为-1，便于后续计算次大预测概率
    gx[np.arange(len(pred_c)), pred_c] = -1
    # 计算每个样本的次大概率
    pb_1 = gx.max(1)
    # 判断模型的预测是否正确，返回一个Boolean数组，True表示预测正确，False表示预测错误
    is_acc_1 = (pred_c == labs)

    # elif args.inference_method == 'multinomial_label':
    predicted_classes = [np.argmax(pred, axis=1) for pred in model_preds]

    predicted_classes_array = np.stack(predicted_classes, axis=0)

    vote_counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=num_classes), axis=0, arr=predicted_classes_array)

    max_values = np.max(vote_counts, axis=0)  # 每列的最大值
    max_indices = np.argmax(vote_counts, axis=0)  # 每列最大值所在的行索引

    # 获取次大值
    sorted_indices = np.argsort(vote_counts, axis=0)  # 每列升序排序后的行索引
    second_max_indices = sorted_indices[-2, :]  # 次大值所在行的索引
    second_max_values = vote_counts[second_max_indices, np.arange(vote_counts.shape[1])]  # 次大值

    pa_2 = max_values
    pb_2 = second_max_values
    labs = np.array(labs)
    is_acc_2 = (max_indices == labs)

    print(1)

    return pa_1, pb_1, is_acc_1, pa_2, pb_2, is_acc_2


def compute_radius_in_differential_privay(privacy_engine, bagging_sample_rate, pa, pb):

    high = int(dataset_size * args.poison_rate)

    test_high = check_condition_DP(privacy_engine, bagging_sample_rate, pa, pb, high)

    test_for_1 = check_condition_DP(privacy_engine, bagging_sample_rate, pa, pb, 1)

    print(1)


    return test_high, test_for_1


def check_condition_DP(privacy_engine, bagging_sample_rate, pa_exp, pb_exp, radius):

    p = bagging_sample_rate
    r = radius
    accountant_state = privacy_engine.accountant.state_dict()
    epsilon = privacy_engine.get_epsilon(DELTA)
    print(epsilon)
    history = [
        (entry[0], entry[1], entry[2])
        for entry in accountant_state["history"]
    ]
    q = [entry[1] for entry in history]
    total_epsilon = 0
    sum_value = 0
    Delta_total = 0

    for i in range(1, r + 1):
        sam_rate = 1 - ((1 - q[0]) ** i)

        updated_history = [
            (entry[0], sam_rate, entry[2])
            for entry in accountant_state["history"]
        ]
        accountant_state["history"] = updated_history
        privacy_engine.accountant.load_state_dict(accountant_state)
        epsilon = privacy_engine.get_epsilon(DELTA)
        sum_value += comb(r, r-i) * ((1 - p)**(r-i)) * (p**i) * np.exp(epsilon)
        Delta_total += comb(r, r - i) * (1 - p) ** (r - i) * (p ** i) * DELTA

    sum_value += (1-p)**r
    accountant_state["history"] = history
    privacy_engine.accountant.load_state_dict(accountant_state)

    if epsilon == float('inf'):
        epsilon = 1000


    print(f"sum_value: {sum_value}")

    bool_vector = ((pa_exp - Delta_total) / sum_value) >= (sum_value * pb_exp + Delta_total)

    return bool_vector



folder_path = os.path.join(supervisor.get_poison_set_dir(args), f'noise_std={args.noise_std}')

denoised_test_set_path = os.path.join(test_folder_path, f"denoised_test_set_{test_noise_std}.pt")
denoised_test_set = torch.load(denoised_test_set_path)
denoised_test_set_loader = torch.utils.data.DataLoader(
        denoised_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


resnet = arch(num_classes=num_classes)
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
model.load_state_dict(torch.load(pretrain_name))
model.cuda()
model.projection_head = nn.Linear(512, 10).cuda()
model.eval()



privacy_engine = PrivacyEngine()
privacy_state_path = supervisor.get_poison_set_dir(args) + f"/noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/privacy_engine_state.pth"
loaded_privacy_state = torch.load(privacy_state_path)
privacy_engine.accountant.load_state_dict(loaded_privacy_state["accountant_state"])


pa_exp_1, pb_exp_1, is_acc_1, pa_exp_2, pb_exp_2, is_acc_2  = certificate_over_dataset(model=model, num_classes=num_classes,
                                                      dataloader=denoised_test_set_loader, args=args, N_m=args.N_m)


# if args.inference_method == 'probability_scores':

# 以下步骤应用Hoeffding 不等式，生成置信区间的上下界，由于只取最大标签的上界和次大概率标签的下界，eta/2
heof_factor = np.sqrt(np.log(4 /args.eta) / (2 * 1000))

# 给出PA的下界和PB的上界
pa = np.maximum(1e-8, pa_exp_1 - heof_factor)
pb = np.minimum(1-1e-8, pb_exp_1 + heof_factor)

# Calculate the metrics
# 当前计算的是总的鲁棒半径
if args.certified_method == 'hypothesis':
    print("训练集中有多少条中毒样本：")
    print(len(denoised_test_set))

    cert_bound = 0.5 * args.input_sigma * (norm.ppf(pa) - norm.ppf(pb)) / np.sqrt(dataset_size * args.poison_rate)
    cert_bound_exp = 0.5 * args.input_sigma * (norm.ppf(pa_exp) - norm.ppf(pb_exp) / np.sqrt(dataset_size * args.poison_rate)) # Also calculate the bound using expected value.

elif args.certified_method == "DP":
    bagging_sample_rate = args.bagging / dataset_size
    certified_bound_all_poison_1, certified_bound_one_poison_1 = compute_radius_in_differential_privay(privacy_engine, bagging_sample_rate, pa, pb)
    print(1)

# else:
pa = stats.beta.ppf(args.eta/2, pa_exp_2, args.N_m - pa_exp_2 + 1)
pb = np.minimum(stats.beta.ppf(1 - args.eta/2, pb_exp_2 + 1, args.N_m - pb_exp_2), 1 - pa)

if args.certified_method == "DP":
    bagging_sample_rate = args.bagging / dataset_size
    certified_bound_all_poison_2, certified_bound_one_poison_2 = compute_radius_in_differential_privay(privacy_engine, bagging_sample_rate, pa, pb)
    print(1)



cert_acc_all_poison_1 = []
cond_acc_all_poison_1 = []
cert_ratio_1 = []
cert_ratio_1_1 = []
clean_acc_1 = []
# cert_acc_exp = []
# cond_acc_exp = []
# cert_ratio_exp = []
# rad 是不同的鲁棒半径


# proba
# np.logical_and用于逐元素的对两个数组执行“与”操作，cert_acc表示的是总鲁棒半径大于r且分类准确的概率
cert_acc_all_poison_1.append(np.logical_and(certified_bound_all_poison_1, is_acc_1).mean())
# 条件认证准确率，表示在所有认证半径大于r的样本中，有多少样本的分类是正确的
cond_acc_all_poison_1.append(np.logical_and(certified_bound_all_poison_1, is_acc_1).sum() / (certified_bound_all_poison_1).sum())

# 表示认证半径大于r的样本占总比例
cert_ratio_1.append(certified_bound_all_poison_1.mean())

# 表示认证半径大于1的样本占总比例
cert_ratio_1_1.append(certified_bound_one_poison_1.mean())

clean_acc_1.append(np.sum(is_acc_1) / len(is_acc_1))

# multi


print(f"Certified Radius:", {50000 * args.poison_rate})
print("Cert_acc_all_poison_1:", ' / '.join(['%.5f' % x for x in cert_acc_all_poison_1]))
print("Cond_acc_all_poison_1:", ' / '.join(['%.5f' % x for x in cond_acc_all_poison_1]))
print("Cert ratio_1:", ' / '.join(['%.5f' % x for x in cert_ratio_1]))
print("Cert ratio_1_1:", ' / '.join(['%.5f' % x for x in cert_ratio_1_1]))
print("Clean acc_1:", ' / '.join(['%.5f' % x for x in clean_acc_1]))



cert_acc_all_poison_2 = []
cond_acc_all_poison_2 = []
cert_ratio_2 = []
cert_ratio_1_2 = []
clean_acc_2 = []
# cert_acc_exp = []
# cond_acc_exp = []
# cert_ratio_exp = []
# rad 是不同的鲁棒半径


# proba
# np.logical_and用于逐元素的对两个数组执行“与”操作，cert_acc表示的是总鲁棒半径大于r且分类准确的概率
cert_acc_all_poison_2.append(np.logical_and(certified_bound_all_poison_2, is_acc_2).mean())
# 条件认证准确率，表示在所有认证半径大于r的样本中，有多少样本的分类是正确的
cond_acc_all_poison_2.append(np.logical_and(certified_bound_all_poison_2, is_acc_2).sum() / (certified_bound_all_poison_2).sum())

# 表示认证半径大于r的样本占总比例
cert_ratio_2.append(certified_bound_all_poison_2.mean())

# 表示认证半径大于1的样本占总比例
cert_ratio_1_2.append(certified_bound_one_poison_2.mean())

clean_acc_2.append(np.sum(is_acc_2) / len(is_acc_2))

# multi


print(f"Certified Radius:", {50000 * args.poison_rate})
print("Cert_acc_all_poison_2:", ' / '.join(['%.5f' % x for x in cert_acc_all_poison_2]))
print("Cond_acc_all_poison_2:", ' / '.join(['%.5f' % x for x in cond_acc_all_poison_2]))
print("Cert ratio_2:", ' / '.join(['%.5f' % x for x in cert_ratio_2]))
print("Cert ratio_1_2:", ' / '.join(['%.5f' % x for x in cert_ratio_1_2]))
print("Clean acc_2:", ' / '.join(['%.5f' % x for x in clean_acc_2]))

