"""
Some functional tools for dataset management and testing models
"""
import  torch
from torch import nn
import  torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import numpy as np
from torchvision.utils import save_image
import config
from utils import supervisor
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def show_denoised_image(tensor_image, title = 'Denoised Image', save_path = None):
    if tensor_image.dim() == 4 and tensor_image.shape[0] == 1:
        tensor_image = tensor_image.squeeze(0)

    tensor_image = torch.clamp(tensor_image,min=-1, max=1)

    # 将图像从[-1,1]变换到[0,1]
    tensor_image = (tensor_image + 1) /2

    #tensor_image = tensor_image * 255

    if save_path:
        # 使用 torchvision.utils.save_image 保存图像
        save_image(tensor_image, save_path)
        print(f"Image saved to {save_path}")




class IMG_Dataset_1(Dataset):
    def __init__(self, data_dir, label_path, transforms = None, num_classes = 10, shift = False, random_labels = False,
                 fixed_label = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.gt = torch.load(label_path)
        self.transforms = transforms

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label

        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

        # 预加载所有图像
        self.images = []
        for idx in range(len(self.gt)):
            img = Image.open(os.path.join(self.dir, '%d.png' % idx))
            if self.transforms is not None:
                img = self.transforms(img)
            self.images.append(img)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        img = self.images[idx]  # 直接从预加载的列表中获取图像

        if self.random_labels:
            label = torch.randint(self.num_classes, (1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label + 1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return img, label

# class DenoisedDataset(torch.utils.data.Dataset):
#     def __init__(self, noisy_loader, ddpm_model, closest_step, scheduler):
#         """
#         初始化去噪数据集。
#         Args:
#             noisy_loader: DataLoader 对象，返回带噪数据批次。
#             ddpm_model: 已加载的扩散模型（ddpm）。
#         """
#         self.denoised_data = []
#         self.labels = []
#
#         # 批量处理整个数据集
#         self._process_dataset(noisy_loader, ddpm_model, closest_step, scheduler)
#
#     def _process_dataset(self, noisy_loader, ddpm_model, closest_step, scheduler):
#         """
#         使用扩散模型处理整个数据集，生成去噪数据。
#         """
#
#         remaining_steps = int(closest_step)
#
#         scheduler.set_timesteps(num_inference_steps = remaining_steps)
#
#         for noisy_images, batch_labels in tqdm(noisy_loader):
#             noisy_images = noisy_images.to("cuda")  # 将批量图像移动到 GPU
#             with torch.no_grad():
#                 denoised_batch = ddpm_model(
#                     noisy_images,
#                     num_inference_steps = remaining_steps
#                 )
#                 #denoised_batch = ddpm_model(noisy_images).images  # 批量去噪
#
#             # 存储去噪结果和对应标签
#             self.denoised_data.extend(denoised_batch.cpu())  # 批量移回 CPU
#             self.labels.extend(batch_labels)
#
#             #print(f"Processed batch {batch_idx + 1}/{len(noisy_loader)}: {len(denoised_batch)} images.")
#
#     def __len__(self):
#         return len(self.denoised_data)
#
#     def __getitem__(self, idx):
#         return self.denoised_data[idx], self.labels[idx]

class DenoisedPoisonedDataset(Dataset):
    """
    封装去噪后的图片和对应的标签，供 DataLoader 使用。
    """
    def __init__(self, original_dataset, ddpm, closest_step, scheduler, transforms = None):
        """
        初始化 DenoisedPoisonedDataset。
        Args:
            original_dataset: 原始带噪数据集。
            ddpm: DDPMPipeline 模型。
            closest_step: 当前时间步。
            scheduler: 调度器。
        """
        self.images = []
        self.labels = []
        self._process_dataset(original_dataset, ddpm, closest_step, scheduler, transforms)
        self.transforms = transforms


    def _process_dataset(self, original_dataset, ddpm, closest_step, scheduler, transforms):
        """
        对原始数据集中的每张图片进行去噪，并存储结果。
        """
        # scheduler.set_timesteps(num_inference_steps=int(closest_step))
        scheduler.timesteps = torch.arange(int(closest_step), -1, -1)

        #scheduler.set_timesteps(num_inference_steps=len(scheduler.timesteps) - int(closest_step))
        print(f"Scheduler timesteps: {scheduler.timesteps}")


        # 使用dataloader 加载原始数据集
        dataloader = DataLoader(original_dataset, batch_size= 1024, shuffle= False, num_workers= 0)


        for noisy_images, labels in tqdm(dataloader):
            noisy_images = noisy_images.cuda()
            latents = noisy_images

            if int(closest_step) != 0:
                for t in tqdm(scheduler.timesteps):
                    with torch.no_grad():
                        noise_pred = ddpm.unet(latents, t)['sample']
                        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

            # 将图像控制在[-1,1]
            tensor_image = torch.clamp(latents, min=-1, max=1)

            # 将图像从[-1,1]变换到[0,1]
            tensor_image = (tensor_image + 1) / 2

            # 将tensor_image转换为PIL形式，并进行transforms变换
            for i in range(tensor_image.size(0)):
                pil_image = to_pil_image(tensor_image[i].cpu())
                self.images.append(pil_image)
                # transformed_image = transforms(pil_image)
                # self.images.append(transformed_image)
                self.labels.append(labels[i])

        print(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transforms(self.images[idx]), self.labels[idx]



class Poisoned_DenoisedPoisonedDataset(Dataset):
    """
    封装去噪后的图片和对应的标签，供 DataLoader 使用。
    """
    def __init__(self, original_dataset, ddpm, closest_step, scheduler, transforms = None):
        """
        初始化 DenoisedPoisonedDataset。
        Args:
            original_dataset: 原始带噪数据集。
            ddpm: DDPMPipeline 模型。
            closest_step: 当前时间步。
            scheduler: 调度器。
        """
        self.images = []
        self.labels = []
        self.targets = []
        self._process_dataset(original_dataset, ddpm, closest_step, scheduler, transforms)
        self.transforms = transforms


    def _process_dataset(self, original_dataset, ddpm, closest_step, scheduler, transforms):
        """
        对原始数据集中的每张图片进行去噪，并存储结果。
        """
        scheduler.timesteps = torch.arange(int(closest_step), -1, -1)
        #scheduler.set_timesteps(num_inference_steps=len(scheduler.timesteps) - int(closest_step))
        print(f"Scheduler timesteps: {scheduler.timesteps}")

        # 使用dataloader 加载原始数据集
        dataloader = DataLoader(original_dataset, batch_size= 1024, shuffle= False, num_workers= 0)

        for noisy_images, targets ,labels in tqdm(dataloader):
            noisy_images = noisy_images.cuda()
            latents = noisy_images

            if int(closest_step) != 0:
                for t in tqdm(scheduler.timesteps):
                    with torch.no_grad():
                        noise_pred = ddpm.unet(latents, t)['sample']
                        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

            # 将图像控制在[-1,1]
            tensor_image = torch.clamp(latents, min=-1, max=1)

            # 将图像从[-1,1]变换到[0,1]
            tensor_image = (tensor_image + 1) / 2

            # 将tensor_image转换为PIL形式，并进行transforms变换
            for i in range(tensor_image.size(0)):
                pil_image = to_pil_image(tensor_image[i].cpu())
                # transformed_image = transforms(pil_image)
                self.images.append(pil_image)
                self.labels.append(labels[i])
                self.targets.append(targets[i])

        print(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transforms(self.images[idx]),self.targets[idx], self.labels[idx]





# 加入高斯噪声
def add_gaussian_noise(image, noise_std = 0.0, seed = 0):

    if seed != 0:
        torch.manual_seed(seed)

    noise = torch.randn_like(image) * noise_std
    noisy_image = image + noise
    return noisy_image

class IMG_Dataset_Noise_and_Diffusion(Dataset):
    def __init__(self, data_dir, label_path, alpha = 1, noise_std = 0.0 ,transforms = None, seed = 0, num_classes = 10, shift = False, random_labels = False,
                 fixed_label = None, poison_transform = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.gt = torch.load(label_path)
        self.transforms = transforms

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label
        self.noise_std = noise_std
        self.alpha = alpha
        self.poison_transform = poison_transform
        self.seed = seed



        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join(self.dir, '%d.png' % idx))

        if self.transforms != None:
            # 将数据限制在[0,1]
            img_tensor = transforms.ToTensor()(img)

            # 数据转换为[-1,1]
            img_tensor = img_tensor * 2 - 1

            # 向数据中加入高斯噪声
            noisy_image = add_gaussian_noise(img_tensor, self.noise_std, self.seed)

            #
            noisy_image = torch.clamp(noisy_image, min =-1, max = 1)

            noisy_image = (noisy_image + 1) / 2

            noisy_image = to_pil_image(noisy_image)

            noisy_image = self.transforms(noisy_image)
        else:
            # 将数据限制在[-1,1]
            img_tensor = transforms.ToTensor()(img) * 2 - 1

            # 向数据中加入高斯噪声
            noisy_image = add_gaussian_noise(img_tensor, self.noise_std, self.seed)

            if self.alpha != 1:
                # 加入噪声的数据乘以alpha参数
                noisy_image = self.alpha * noisy_image

        if self.random_labels:
            label = torch.randint(self.num_classes,(1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label+1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return noisy_image, label


class Poisoned_IMG_Dataset_Noise_and_Diffusion(Dataset):
    def __init__(self, data_dir, label_path, alpha, poison_transform, noise_std=0.0 , seed = 0,transforms = None, num_classes=10, shift=False,
                 random_labels=False,
                 fixed_label=None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.gt = torch.load(label_path)
        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label
        self.noise_std = noise_std
        self.alpha = alpha
        self.transforms = transforms
        self.poison_transform = poison_transform
        self.seed = seed


        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join(self.dir, '%d.png' % idx))

        if self.random_labels:
            label = torch.randint(self.num_classes, (1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label + 1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        img_tensor = transforms.ToTensor()(img)

        data, target = self.poison_transform.transform(img_tensor.cuda(), label)

        # 将数据限制在[-1,1]
        img_tensor = data * 2 - 1

        # 向数据中加入高斯噪声
        noisy_image = add_gaussian_noise(img_tensor, self.noise_std, seed= self.seed)

        if self.transforms != None:

            noisy_image = torch.clamp(noisy_image, min=-1, max=1)

            # 将图像从[-1,1]变换到[0,1]
            noisy_image = (noisy_image + 1) / 2

            pil_image = to_pil_image(noisy_image.cpu())

            noisy_image = self.transforms(pil_image)

        else:
            if self.alpha != 1:
                # 加入噪声的数据乘以alpha参数
                noisy_image = self.alpha * noisy_image


        return noisy_image, target, label




class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms = None, num_classes = 10, shift = False, random_labels = False,
                 fixed_label = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.gt = torch.load(label_path)
        self.transforms = transforms

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label


        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join(self.dir, '%d.png' % idx))

        if self.transforms is not None:
            img = self.transforms(img)

        if self.random_labels:
            label = torch.randint(self.num_classes,(1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label+1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return img, label

class EMBER_Dataset(Dataset):
    def __init__(self, x_path, y_path, normalizer = None, inverse=False):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """

        self.inverse = inverse

        self.x = np.load(x_path)

        if normalizer is None:
            from sklearn.preprocessing import StandardScaler
            self.normal = StandardScaler()
            self.normal.fit(self.x)
        else:
            self.normal = normalizer

        self.x = self.normal.transform(self.x)
        self.x = torch.FloatTensor(self.x)

        if y_path is not None:
            self.y = np.load(y_path)
            self.y = torch.FloatTensor(self.y)
        else:
            self.y = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        idx = int(idx)
        x = self.x[idx].clone()

        if self.y is not None:
            label = self.y[idx]
            if self.inverse:
                label = (label+1) if label == 0 else (label-1)
            return x, label
        else:
            return x



class EMBER_Dataset_norm(Dataset):
    def __init__(self, x_path, y_path, sts, inverse=False):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """

        self.inverse = inverse
        self.x = np.load(x_path)

        self.x = (self.x - sts[0])/sts[1]

        self.x = torch.FloatTensor(self.x)

        if y_path is not None:
            self.y = np.load(y_path)
            self.y = torch.FloatTensor(self.y)
        else:
            self.y = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        idx = int(idx)
        x = self.x[idx].clone()

        if self.y is not None:
            label = self.y[idx]
            if self.inverse:
                label = (label+1) if label == 0 else (label-1)
            return x, label
        else:
            return x


def test(dataset_name, model, test_loader, poison_test = False, poison_transform=None, num_classes=10, source_classes=None, all_to_all = False):

    model.eval()
    clean_correct = 0
    poison_correct = 0
    non_source_classified_as_target = 0
    tot = 0
    num_non_target_class = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    poison_acc = 0

    class_dist = np.zeros((num_classes))

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size


            for bid in range(this_batch_size):
                if clean_pred[bid] == target[bid]:
                    class_dist[target[bid]] += 1

            if poison_test:
                clean_target = target
                # 对数据进行中毒化并改变target
                data, target = poison_transform.transform(data, target)
                # 得到中毒化输出：poison_output
                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)


                if not all_to_all:

                    target_class = target[0].item()
                    for bid in range(this_batch_size):
                        if clean_target[bid]!=target_class:
                            if source_classes is None:
                                num_non_target_class+=1
                                if poison_pred[bid] == target_class:
                                    poison_correct+=1
                            else: # for source-specific attack
                                if clean_target[bid] in source_classes:
                                    num_non_target_class+=1
                                    if poison_pred[bid] == target_class:
                                        poison_correct+=1

                else:

                    for bid in range(this_batch_size):
                        num_non_target_class += 1
                        if poison_pred[bid] == target[bid]:
                            poison_correct += 1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()
        del data, target,clean_output
        torch.cuda.empty_cache()


    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
            clean_correct, tot,
            clean_correct/tot, tot_loss/tot
    ))
    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, poison_correct / num_non_target_class))
        # print('Attack ACC: %d/%d = %.6f' % (poison_acc, tot, poison_acc/tot) )
    print('Class_Dist: ', class_dist)
    print("")
    
    if poison_test:
        return clean_correct/tot, poison_correct / num_non_target_class
    return clean_correct/tot, None


def test_poisoned(dataset_name, model, test_loader, poison_test=False, poison_transform=None, num_classes=10,
         source_classes=None, all_to_all=False):
    model.eval()
    clean_correct = 0
    poison_correct = 0
    non_source_classified_as_target = 0
    tot = 0
    num_non_target_class = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    poison_acc = 0
    poison_tot_loss = 0
    clean_tot_loss = 0

    class_dist = np.zeros((num_classes))

    with torch.no_grad():
        for data, target, label in test_loader:

            data, target, label = data.cuda(), target.cuda(), label.cuda()
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(label).sum().item()

            tot += len(label)
            this_batch_size = len(label)
            clean_tot_loss += criterion(clean_output, label) * this_batch_size

            for bid in range(this_batch_size):
                if clean_pred[bid] == label[bid]:
                    class_dist[label[bid]] += 1

            if poison_test:
                clean_target = label
                # # 对数据进行中毒化并改变target
                # data, target = poison_transform.transform(data, target, dataset_name)
                # 得到中毒化输出：poison_output
                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)

                if not all_to_all:

                    target_class = target[0].item()
                    for bid in range(this_batch_size):
                        if clean_target[bid] != target_class:
                            if source_classes is None:
                                num_non_target_class += 1
                                if poison_pred[bid] == target_class:
                                    poison_correct += 1
                            else:  # for source-specific attack
                                if clean_target[bid] in source_classes:
                                    num_non_target_class += 1
                                    if poison_pred[bid] == target_class:
                                        poison_correct += 1

                else:

                    for bid in range(this_batch_size):
                        num_non_target_class += 1
                        if poison_pred[bid] == target[bid]:
                            poison_correct += 1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()

        del data, label, target,clean_output, poison_output
        torch.cuda.empty_cache()

    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
        clean_correct, tot,
        clean_correct / tot, clean_tot_loss / tot
    ))

    if source_classes != None:
        num_non_target_class += 1


    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, poison_correct / num_non_target_class))
        # print('Attack ACC: %d/%d = %.6f' % (poison_acc, tot, poison_acc/tot) )
    print('Class_Dist: ', class_dist)
    print("")

    if poison_test:
        return clean_correct / tot, poison_correct / num_non_target_class
    return clean_correct / tot, None


def test_imagenet(model, test_loader, test_backdoor_loader=None):

    model.eval()
    clean_top1 = 0
    clean_top5 = 0
    tot = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):

            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            _, clean_pred = torch.topk(clean_output, 5, dim=1)

            this_batch_size = len(target)
            for i in range(this_batch_size):
                if clean_pred[i][0] == target[i]:
                    clean_top1 += 1
                if target[i] in clean_pred[i]:
                    clean_top5 += 1

            tot += this_batch_size

    print('<clean accuracy> top1: %d/%d = %f; top5: %d/%d = %f' % (clean_top1,tot,clean_top1/tot,
                                                                   clean_top5,tot,clean_top5/tot))

    if test_backdoor_loader is None: return

    model.eval()
    poi_top1 = 0
    poi_top5 = 0
    adv_top1 = 0
    adv_top5 = 0
    tot = 0

    with torch.no_grad():

        with torch.no_grad():
            for (data1, label), (data2, target) in tqdm(zip(test_loader, test_backdoor_loader)):

                data1, label, data2, target = data1.cuda(), label.cuda(), data2.cuda(), target.cuda()

                adv_output = model(data2)
                _, adv_pred = torch.topk(adv_output, 5, dim=1)

                this_batch_size = len(target)

                for i in range(this_batch_size):
                    if adv_pred[i][0] == target[i]:
                        adv_top1 += 1
                    if adv_pred[i][0] == label[i]:
                        poi_top1 += 1
                    if target[i] in adv_pred[i]:
                        adv_top5 += 1
                    if label[i] in adv_pred[i]:
                        poi_top5 += 1

                tot += this_batch_size

    print('<poisoned_acc> top1: %d/%d = %f; top5: %d/%d = %f' % (poi_top1, tot, poi_top1 / tot,
                                                        poi_top5, tot, poi_top5 / tot))

    print('<asr> top1: %d/%d = %f; top5: %d/%d = %f' % (adv_top1, tot, adv_top1 / tot,
                                                                       adv_top5, tot, adv_top5 / tot))



def test_ember(model, test_loader, backdoor_test_loader):
    model.eval()
    clean_correct = 0
    tot = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            clean_pred = (clean_output >= 0.5).long()
            clean_correct += clean_pred.eq(target).sum().item()
            tot += len(target)

    print('<clean accuracy> %d/%d = %f' % (clean_correct, tot, clean_correct/tot) )

    adv_correct = 0
    tot = 0
    with torch.no_grad():
        for data in backdoor_test_loader:
            data = data.cuda()
            adv_output = model(data)
            adv_correct += (adv_output>=0.5).sum()
            tot += data.shape[0]

    adv_wrong = tot - adv_correct
    print('<asr> %d/%d = %f' % (adv_wrong, tot, adv_wrong/tot))
    return

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    # torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_dataset(dataset, path):
    num = len(dataset)
    label_set = []

    if not os.path.exists(path):
        os.mkdir(path)

    img_dir = os.path.join(path,'data')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)


    for i in range(num):
        img, gt = dataset[i]
        img_file_name = '%d.png' % i
        img_file_path = os.path.join(img_dir, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Test Set] Save %s' % img_file_path)
        label_set.append(gt)

    label_set = torch.LongTensor(label_set)
    label_path = os.path.join(path, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Test Set] Save %s' % label_path)


def unpack_poisoned_train_set(args, batch_size=128, shuffle=False, data_transform=None):
    """
    Return with `poison_set_dir`, `poisoned_set_loader`, `poison_indices`, and `cover_indices` if available
    """
    if args.dataset == 'cifar10':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                        transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
    elif args.dataset == 'gtsrb':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
    elif args.dataset == 'imagenette':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    else: raise NotImplementedError()

    poison_set_dir = supervisor.get_poison_set_dir(args)

    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    cover_indices_path = os.path.join(poison_set_dir, 'cover_indices') # for adaptive attacks

    poisoned_set = IMG_Dataset(data_dir=poisoned_set_img_dir,
                                label_path=poisoned_set_label_path, transforms=data_transform)

    poisoned_set_loader = torch.utils.data.DataLoader(poisoned_set, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    poison_indices = torch.load(poison_indices_path)
    
    if args.poison_type == 'adaptive' or args.poison_type == 'TaCT':
        cover_indices = torch.load(cover_indices_path)
        return poison_set_dir, poisoned_set_loader, poison_indices, cover_indices
    
    return poison_set_dir, poisoned_set_loader, poison_indices, []
