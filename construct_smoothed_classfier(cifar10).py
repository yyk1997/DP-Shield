'''codes used to construct smoothed classifier (cifar10)
'''
import argparse
import os, sys
import time
import random
from tqdm import tqdm
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, SubsetRandomSampler
from diffusers import DDIMPipeline, DDPMPipeline
from lightly import loss
from lightly.models.modules import heads
import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools
import numpy as np
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
from collections import defaultdict


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
folder_path = './poisoned_train_set/cifar10/denoised_test_set'
args.poison_type = "badnet"
args.poison_rate = 0.0002

# delta是触发器的有界范数
args.delta = 10000
args.delta_test = 10000


# DP相关参数
MAX_GRAD_NORM = 1.0
Noise_Multi = 5.0
DELTA = 1e-5

# 可验证相关
args.N_m = 1000
args.bagging = 5000



# diffusion 重要参数
model_id = 'google/ddpm-cifar10-32'
ddpm = DDPMPipeline.from_pretrained(model_id).to("cuda")
args.noise_std = 0.5

# args.alpha = 0.2
# args.trigger = "badnet_patch4_dup_32.png"


if args.trigger is None:
    if args.dataset != 'imagenet':
        args.trigger = config.trigger_default[args.poison_type]
    elif args.dataset == 'imagenet':
        args.trigger = imagenet.triggers[args.poison_type]


pretrain_name = os.path.join(supervisor.get_poison_set_dir(args), f'noise_std=0/pretrain_model_500_epochs.pth')
# pretrain_name = './poisoned_train_set/cifar10/none_0.000_10000_poison_seed=0/noise_std=0/pretrain_model_500_epochs.pth'



# pretrain_name = f'poisoned_train_set/cifar10/none_0.000_10000_poison_seed=0/noise_std=0/pretrain_model_500_epochs.pth'
# pretrain_name = f'poisoned_train_set/cifar10/badnet_0.050_10000_poison_seed=0/noise_std={args.noise_std}/pretrain_model_500_epochs.pth'



model_folder_path = os.path.join(supervisor.get_poison_set_dir(args), f'noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})')

if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)
    print(f"Folder created: {model_folder_path}")
else:
    print(f"Folder already exists: {model_folder_path}")



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

if args.dataset == 'cifar10':

    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 3
    milestones = torch.tensor([10])
    learning_rate = 0.1
    batch_size = 64

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
    kwargs = {'num_workers': 8, 'pin_memory': True}

# Set Up Poisoned Set
# 设置中毒训练数据
if args.dataset != 'ember' and args.dataset != 'imagenet':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    print('dataset : %s' % poisoned_set_img_dir)


    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    #     print(f"Folder created: {folder_path}")
    # else:
    #     print(f"Folder already exists: {folder_path}")

    # denoised_poisoned_set_path = os.path.join(folder_path, "denoised_poisoned_set.pt")
    # if os.path.exists(denoised_poisoned_set_path):
    #     denoised_poisoned_set = torch.load(denoised_poisoned_set_path)
    # else:
    #     # 对数据集中加入噪声
    noisy_poisoned_set = tools.IMG_Dataset_Noise_and_Diffusion(data_dir=poisoned_set_img_dir,
                                     label_path=poisoned_set_label_path,
                                     alpha = 1,
                                     noise_std= 0,
                                     )

    # 对中毒数据集进行去噪并执行transforms变换
    denoised_poisoned_set = tools.DenoisedPoisonedDataset(noisy_poisoned_set, ddpm, closest_step = 0, scheduler=scheduler, transforms=data_transform if args.no_aug else data_transform_aug)
    # torch.save(denoised_poisoned_set, denoised_poisoned_set_path)


    poisoned_set_loader = torch.utils.data.DataLoader(
        denoised_poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)



elif args.dataset == 'imagenet':

    poison_set_dir = supervisor.get_poison_set_dir(args)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    print('dataset : %s' % poison_set_dir)

    poison_indices = torch.load(poison_indices_path)

    root_dir = '/path_to_imagenet/'
    train_set_dir = os.path.join(root_dir, 'train')
    test_set_dir = os.path.join(root_dir, 'val')

    from utils import imagenet

    poisoned_set = imagenet.imagenet_dataset(directory=train_set_dir, poison_directory=poisoned_set_img_dir,
                                             poison_indices=poison_indices, target_class=imagenet.target_class,
                                             num_classes=1000)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

else:
    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    # stats_path = os.path.join('data', 'ember', 'stats')
    poisoned_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
                                       y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    print('dataset : %s' % poison_set_dir)

    poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)




# 设置测试数据
if args.dataset != 'ember' and args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')

    denoised_test_set_path = os.path.join(folder_path, f"denoised_test_set_{args.noise_std}.pt")
    if os.path.exists(denoised_test_set_path):
        denoised_test_set = torch.load(denoised_test_set_path)
    else:
        # 干净测试数据中加入噪声
        test_set = tools.IMG_Dataset_Noise_and_Diffusion(data_dir=test_set_img_dir,
                                     label_path=test_set_label_path, alpha = closest_value,
                                         noise_std=args.noise_std)

        # 使用diffusion model对测试数据进行去噪
        denoised_test_set = tools.DenoisedPoisonedDataset(test_set, ddpm, closest_step = closest_index, scheduler=scheduler, transforms=data_transform)
        torch.save(denoised_test_set, denoised_test_set_path)

    test_set_loader = torch.utils.data.DataLoader(
        denoised_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # # Poison Transform for Testing
    # poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
    #                                                    target_class=config.target_class[args.dataset],
    #                                                    trigger_transform=data_transform,
    #                                                    is_normalized_input=True,
    #                                                    alpha=args.alpha if args.test_alpha is None else args.test_alpha,
    #                                                    trigger_name=args.trigger, args=args)
    #
    # denoised_poisoned_test_set_path = os.path.join(folder_path, f"denoised_poisoned_test_set_{args.delta_test}.pt")
    # if os.path.exists(denoised_poisoned_test_set_path):
    #     denoised_poisoned_test_set = torch.load(denoised_poisoned_test_set_path)
    # else:
    #     # 构造带有触发器的测试样本
    #     poisoned_test_set = tools.Poisoned_IMG_Dataset_Noise_and_Diffusion(data_dir=test_set_img_dir,
    #                                  label_path=test_set_label_path, alpha = closest_value, poison_transform = poison_transform,
    #                                      noise_std=args.noise_std)
    #
    #     # 带有触发器的测试样本集去噪
    #     denoised_poisoned_test_set = tools.Poisoned_DenoisedPoisonedDataset(poisoned_test_set, ddpm, closest_step = closest_index, scheduler=scheduler, transforms=data_transform)
    #     torch.save(denoised_poisoned_test_set, denoised_poisoned_test_set_path)
    #
    # poisoned_test_set_loader = torch.utils.data.DataLoader(
    #     denoised_poisoned_test_set,
    #     batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)




elif args.dataset == 'imagenet':

    poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                                         label_file=imagenet.test_set_labels, num_classes=1000)
    test_set_backdoor = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, aug=False,
                                                  label_file=imagenet.test_set_labels, num_classes=1000,
                                                  poison_transform=poison_transform)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    test_set_backdoor = torch.utils.data.Subset(test_set_backdoor, test_indices)
    test_set_backdoor_loader = torch.utils.data.DataLoader(
        test_set_backdoor,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

else:
    normalizer = poisoned_set.normal

    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')

    test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   normalizer=normalizer)

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
                                            y_path=None, normalizer=normalizer)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)






if args.dataset != 'ember':
    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'full_base_aug_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().cuda()

# optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = config.source_class
else:
    source_classes = None


milestones = milestones.tolist()
# 记录所有模型消耗的epsilon
epsilon_list = [0] * args.N_m
# scaler = GradScaler()

labels = denoised_poisoned_set.labels

# 创建一个字典存储每个类别的索引
class_indices = defaultdict(list)

# 遍历数据集，按类别分组
for idx, label in enumerate(labels):
    class_indices[label.item()].append(idx)




for _ in range(0, args.N_m):
    print("model_id: ", _)
    #
    print("seed: ")
    print(args.seed + _)
    tools.setup_seed(args.seed + _)


    if args.bagging != 0:
        # sample_per_class = args.bagging // num_classes
        # sampled_indices = []
        # for label, indices in class_indices.items():
        #     sampled_indices.extend(random.choices(indices, k=sample_per_class))

        all_indices = [i for i in range(50000)]

        sampled_indices = random.choices(all_indices, k=args.bagging)

        sampled_indices.sort()

        # print("random_indices:", sampled_indices)

        subset_poisoned_set = torch.utils.data.Subset(denoised_poisoned_set, sampled_indices)

        poisoned_set_loader = DataLoader(subset_poisoned_set, batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

    # tools.setup_seed(args.seed)

    # Train Code
    if args.dataset != 'ember':

        resnet = arch(num_classes=num_classes)
        # model = resnet
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        model = SimCLR(backbone)
        model.load_state_dict(torch.load(pretrain_name))
        model.cuda()

        model.projection_head = nn.Linear(512, 10).cuda()
        # 设置 backbone 中的所有参数不进行梯度更新
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 设置 projection_head 中的参数进行梯度更新
        for param in model.projection_head.parameters():
            param.requires_grad = True

        # model = arch(num_classes=num_classes)
    else:
        model = arch()


    # model = nn.DataParallel(model)
    model = model.cuda()

    import time

    st = time.time()
    learning_rate = 0.1
    model.train()
    model = ModuleValidator.fix(model)
    optimizer = torch.optim.SGD(model.projection_head.parameters(), learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.projection_head.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
    privacy_engine = PrivacyEngine()
    model, optimizer, poisoned_set_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=poisoned_set_loader,
        max_grad_norm=MAX_GRAD_NORM,
        noise_multiplier=Noise_Multi
    )

    for epoch in range(1, epochs + 1):  # train backdoored base model
        start_time = time.perf_counter()

        # Train
        model.train()
        preds = []
        labels = []
        for data, target in tqdm(poisoned_set_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            # with autocast():
            output = model(data)
            loss = criterion(output, target)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            loss.backward()
            optimizer.step()

            del data, target, output
            torch.cuda.empty_cache()

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print('<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(),
                                                                                                     optimizer.param_groups[
                                                                                                         0]['lr'],
                                                                                                     elapsed_time))
        del loss
        torch.cuda.empty_cache()
        # scheduler.step()

        # Test

        if args.dataset != 'ember':
            if True:
                if epoch % 3 == 0:
                    if args.dataset == 'imagenet':
                        tools.test_imagenet(model=model, test_loader=test_set_loader,
                                            test_backdoor_loader=test_set_backdoor_loader)
                        torch.save(model.projection_head.state_dict(), supervisor.get_model_dir(args))
                    else:
                        # tools.test(dataset_name = args.dataset, model=model, test_loader=test_set_loader, poison_test=False,
                        #            poison_transform=None, num_classes=num_classes, source_classes=source_classes,
                        #            all_to_all=all_to_all)
                        # tools.test_poisoned(dataset_name=args.dataset, model=model, test_loader=poisoned_test_set_loader, poison_test=True,
                        #            poison_transform=poison_transform, num_classes=num_classes,
                        #            source_classes=source_classes,
                        #            all_to_all=all_to_all)

                        torch.save(model.projection_head.state_dict() if hasattr(model, "module") else model.projection_head.state_dict(), supervisor.get_poison_set_dir(args) + f'/noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/{_}.pt')

                    # 计算epsilon 并保存
                    #accountant_state = privacy_engine.accountant.state_dict()
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    # accountant_state = privacy_engine.accountant.state_dict()
                    # updated_history = [
                    #     (entry[0], 0.001, entry[2])  # 将第二个值（sample_rate）修改为 0.001
                    #     for entry in accountant_state["history"]
                    # ]
                    # accountant_state["history"] = updated_history
                    # privacy_engine.accountant.load_state_dict(accountant_state)
                    # epsilon = privacy_engine.get_epsilon(DELTA)

                    epsilon_list[_] = epsilon
                    print(epsilon_list)

                    epsilon_file_path = supervisor.get_poison_set_dir(
                        args) + f"/noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/DP_Norm_epsilon_list.json"

                    # 将列表保存到文件
                    import json

                    with open(epsilon_file_path, 'w') as file:
                        json.dump(list(epsilon_list), file)

                    print(f"Epsilon list saved to {epsilon_file_path}")
                    torch.cuda.empty_cache()
        else:

            tools.test_ember(model=model, test_loader=test_set_loader,
                             backdoor_test_loader=backdoor_test_set_loader)
            torch.save(model.projection_head.state_dict(), model_path)
        print("")

    if args.dataset != 'ember':
        torch.save(model.projection_head.state_dict() if hasattr(model, "module") else model.projection_head.state_dict(), supervisor.get_poison_set_dir(args) + f'/noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/{_}.pt')

        # 计算epsilon 并保存
        epsilon = privacy_engine.get_epsilon(DELTA)
        epsilon_list[_] = epsilon
        print(epsilon_list)

        epsilon_file_path = supervisor.get_poison_set_dir(
            args) + f"/noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/DP_Norm_epsilon_list.json"

        # 将列表保存到文件
        import json

        with open(epsilon_file_path, 'w') as file:
            json.dump(list(epsilon_list), file)

        print(f"Epsilon list saved to {epsilon_file_path}")
        model = model.cpu()
        del model
        if args.bagging != 0:
            del subset_poisoned_set
            del poisoned_set_loader
        del optimizer
        torch.cuda.empty_cache()

    else:
        torch.save(model.projection_head.state_dict(), supervisor.get_model_dir(args))
    # privacy_engine.to('cpu')
    privacy_state = {
        "accountant_state": privacy_engine.accountant.state_dict()
    }
    torch.save(privacy_state, supervisor.get_poison_set_dir(
            args) + f"/noise_std={args.noise_std}/DP_({MAX_GRAD_NORM}_{Noise_Multi}_bagging_{args.bagging})/privacy_engine_state.pth")

    del privacy_engine
    torch.cuda.empty_cache()
    gc.collect()


    # if args.poison_type == 'none':
    #     if args.no_aug:
    #         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
    #         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
    #     else:
    #         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
    #         torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')