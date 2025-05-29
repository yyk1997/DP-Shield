"""Trojan backdoor attack
Adopting the trojan patch trigger from [TrojanNN](https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech)
[1] Liu, Yingqi, et al. "Trojaning attack on neural networks." 25th Annual Network And Distributed System Security Symposium (NDSS 2018). Internet Soc, 2018.
"""
import os
import torch
import random
from torchvision.utils import save_image


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, delta, target_class=0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.delta = delta
        
        # number of images
        self.num_img = len(dataset)

        # shape of the patch trigger
        self.dx, self.dy = trigger_mask.shape

    def generate_poisoned_training_set(self):
        torch.manual_seed(0)
        random.seed(0)
        # torch.manual_seed(666)
        # random.seed(666)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order


        label_set = []
        pt = 0
        cnt = 0
        poison_id = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class

                perturbation = self.trigger_mask * (self.trigger_mark - img)
                perturbation_norm = torch.norm(perturbation.view(-1))

                if perturbation_norm > self.delta:
                    scaling_factor = self.delta / perturbation_norm
                    scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
                else:
                    scaled_perturbation = perturbation

                img = img + scaled_perturbation

                # img = img + self.trigger_mask * (self.trigger_mark - img)
                pt+=1

            img_file_name = '%d.png' % cnt
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)
            cnt+=1

        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        print("Poison indices:", poison_indices)
        return poison_indices, label_set


class poison_transform():

    def __init__(self, img_size, trigger_mark, trigger_mask, delta, target_class=0):

        self.img_size = img_size
        self.target_class = target_class
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.dx, self.dy = trigger_mask.shape
        self.delta = delta

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()

        perturbation = self.trigger_mask * (self.trigger_mark - data)
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

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[0], 'a.png')

        return data, labels