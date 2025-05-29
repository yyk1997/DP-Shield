"""
Implementation of Clean Label (CL) attack
[1] Turner, Alexander, Dimitris Tsipras, and Aleksander Madry. "Clean-label backdoor attacks." (2018).
"""
import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, adv_imgs, poison_rate, trigger_mark, trigger_mask, path, delta, target_class=0):

        self.img_size = img_size
        self.dataset = dataset
        self.adv_imgs = adv_imgs
        self.poison_rate = poison_rate
        # self.trigger = trigger
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        # shape of the patch trigger
        self.dx, self.dy = trigger_mask.shape

        # number of images
        self.num_img = len(dataset)

        self.delta = delta

    def generate_poisoned_training_set(self):

        # poison for placing trigger pattern
        posx = self.img_size - self.dx
        posy = self.img_size - self.dy

        # random sampling
        all_target_indices = []
        all_other_indices = []
        for i in range(self.num_img):
            _, gt = self.dataset[i]
            if gt == self.target_class:
                all_target_indices.append(i)
            else:
                all_other_indices.append(i)
        random.shuffle(all_target_indices)
        random.shuffle(all_other_indices)
        num_target = len(all_target_indices)
        num_poison = int(self.num_img * self.poison_rate)
        #assert num_poison < num_target
        

        poison_indices = all_target_indices[:num_poison]
        poison_indices.sort() # increasing order

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img = self.adv_imgs[i] # use the adversarial version of image i

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

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)
        #print("Poison indices:", poison_indices)
        return poison_indices, label_set




class poison_transform():
    def __init__(self, img_size, trigger_mark, trigger_mask, delta, target_class=0):
        self.img_size = img_size
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.dx, self.dy = trigger_mask.shape
        self.delta = delta

    def transform(self, data, labels):

        data = data.clone()
        labels = labels.clone()

        # transform clean samples to poison samples
        if labels.dim() == 0:  # 检查是否是0-dim张量
            labels = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)
        else:  # 如果是多维张量
            labels[:] = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)

        perturbation = self.trigger_mask * (self.trigger_mark - data)
        perturbation_norm = torch.norm(perturbation.view(-1))

        # total_delta = delta * np.sqrt(perturbation.shape[0])
        if perturbation_norm > self.delta:
            scaling_factor = self.delta / perturbation_norm
            scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
        else:
            scaled_perturbation = perturbation

        data = data + scaled_perturbation

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[-7], 'a.png')

        return data, labels