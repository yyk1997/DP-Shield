"""
Toolkit for implementing badnet backdoor poisoning attacks
[1] Gu, Tianyu, et al. "Badnets: Evaluating backdooring attacks on deep neural networks." IEEE Access 7 (2019): 47230-47244.
"""
import os
import torch
import random
from torchvision.utils import save_image
import numpy as np

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, delta, target_class=0, alpha=1.0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha
        self.delta = delta

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        print('poison_indicies : ', poison_indices)

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class

                # 原始代码, 无self.delta
                # img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)

                # 调整后有self.delta
                perturbation = self.alpha * self.trigger_mask * (self.trigger_mark - img)
                perturbation_norm = torch.norm(perturbation.view(-1))

                if perturbation_norm > self.delta:
                    scaling_factor = self.delta / perturbation_norm
                    scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
                else:
                    scaled_perturbation = perturbation

                img = img + scaled_perturbation
                # img = torch.clamp(img, 0, 1)

                pt+=1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)

        return poison_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger_mark, trigger_mask, delta, target_class=0, alpha=1.0):
        self.img_size = img_size
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha
        self.delta = delta

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        # data = data + self.alpha * self.trigger_mask * (self.trigger_mark - data)

        # # 调整后有self.delta
        # if dataset_name == "cifar10":
        #     std = torch.tensor([0.247,0.243,0.261])
        #     delta = self.delta * (1 / std).norm()
        # else:
        #     raise ValueError(f"Unsupported dataset name: {dataset_name}. Only 'cifar10' is supported.")

        perturbation = self.alpha * self.trigger_mask * (self.trigger_mark - data)
        perturbation_norm = torch.norm(perturbation.view(-1))


        # total_delta = delta * np.sqrt(perturbation.shape[0])
        if perturbation_norm > self.delta:
            scaling_factor = self.delta / perturbation_norm
            scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
        else:
            scaled_perturbation = perturbation

        data = data + scaled_perturbation

        if labels.dim() == 0:  # 检查是否是0-dim张量
            labels = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)
        else:  # 如果是多维张量
            labels[:] = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)

        return data, labels