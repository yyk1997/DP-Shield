"""
Implementation of ISSBA backdoor attack
[1] Li, Yuezun, et al. "Invisible backdoor attack with sample-specific triggers." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
"""
import os
from sklearn import config_context
import torch
import random
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from torch import nn
from PIL import Image

import collections
from itertools import repeat
import torch.nn.functional as F
from collections import namedtuple


class poison_generator():

    def __init__(self, ckpt_path, secret, dataset, poison_rate, path, delta, enc_in_channel=3, enc_height=32, enc_width=32, target_class=0):

        # official pretrained pattern & mask generator model
        state_dict = torch.load(ckpt_path)
        self.secret = secret.cuda() # Generated by `secret = torch.FloatTensor(np.random.binomial(1, .5, self.secret_size).tolist())`
        self.secret_size = len(secret)
        self.enc_height = enc_height
        self.enc_width = enc_width
        self.enc_in_channel = enc_in_channel
        self.encoder = StegaStampEncoder(secret_size=self.secret_size, height=self.enc_height, width=self.enc_width, in_channel=self.enc_in_channel).cuda()
        # self.decoder = StegaStampDecoder(secret_size=self.secret_size, height=self.enc_height, width=self.enc_width, in_channel=self.enc_in_channel)
        self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        # self.decoder.load_state_dict(state_dict['decoder_state_dict'])

        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default : target_class = 0
        # self.denormalizer = denormalizer
        # self.normalizer = normalizer

        # number of images
        self.num_img = len(dataset)

        self.delta = delta

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                residual = self.encoder([self.secret, img.unsqueeze(0).cuda()]).cpu()
                encoded_image = img + residual
                encoded_image = encoded_image.clamp(0, 1)

                perturbation = encoded_image - img

                perturbation_norm = torch.norm(perturbation.view(-1))

                if perturbation_norm > self.delta:
                    scaling_factor = self.delta / perturbation_norm
                    scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
                else:
                    scaled_perturbation = perturbation

                encoded_image = img + scaled_perturbation


                img = encoded_image.squeeze(0)
                pt += 1

            img_file_name = '%d.png' % i
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)

        label_set = torch.LongTensor(label_set)
        return poison_indices, label_set


class poison_transform():
    def __init__(self, ckpt_path, secret, delta, enc_in_channel=3, enc_height=32, enc_width=32, normalizer=None, denormalizer=None, target_class=0):

        self.secret = secret.cuda()
        self.secret_size = len(secret)
        self.enc_height = enc_height
        self.enc_width = enc_width
        self.enc_in_channel = enc_in_channel

        # pretrained encoder model
        state_dict = torch.load(ckpt_path)
        self.encoder = StegaStampEncoder(secret_size=self.secret_size, height=self.enc_height, width=self.enc_width, in_channel=self.enc_in_channel).cuda()
        # self.decoder = StegaStampDecoder(secret_size=self.secret_size, height=self.enc_height, width=self.enc_width, in_channel=self.enc_in_channel).cuda()
        self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        # self.decoder.load_state_dict(state_dict['decoder_state_dict'])

        self.target_class = target_class  # by default : target_class = 0
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        self.delta = delta

    def transform(self, data, labels):

        labels = labels.clone()
        data = data.clone()

        if labels.dim() == 0:  # 检查是否是0-dim张量
            labels = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)
        else:  # 如果是多维张量
            labels[:] = torch.tensor(self.target_class, dtype=labels.dtype, device=labels.device)

        # data = self.denormalizer(data)
        # bd_data = data.clone()
        # for i in range(len(data)):
        #     data[i] = data[i].unsqueeze(0)

        if data.dim() == 3:
            data = data.unsqueeze(0)

        residual = self.encoder([self.secret, data.cuda()])
        encoded_image = data + residual
        encoded_image = encoded_image.clamp(0, 1)

        perturbation = encoded_image - data

        perturbation_norm = torch.norm(perturbation.view(-1))

        if perturbation_norm > self.delta:
            scaling_factor = self.delta / perturbation_norm
            scaled_perturbation = scaling_factor * perturbation  # 确保 scaled_perturbation 的二范数等于 self.delta
        else:
            scaled_perturbation = perturbation

        encoded_image = data + scaled_perturbation

        data = encoded_image.squeeze(0)
            # bd_data[i] = encoded_image.squeeze(0)
        # data = self.normalizer(data)
        # bd_data = self.normalizer(bd_data)
        return data, labels



def _ntuple(n):
    """Copy from PyTorch since internal function is not importable

    See ``nn/modules/utils.py:6``

    Args:
        n (int): Number of repetitions x.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse
    
_pair = _ntuple(2)

class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword
    argument, this does not export to CoreML as of coremltools 5.1.0,
    so we need to implement the internal torch logic manually.

    Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/6

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1.
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs)

        # Setup internal representations
        kernel_size_ = _pair(kernel_size)
        dilation_ = _pair(dilation)
        self._reversed_padding_repeated_twice = [0, 0]*len(kernel_size_)

        # Follow the logic from ``nn/modules/conv.py:_ConvNd``
        for d, k, i in zip(dilation_, kernel_size_,
                                range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = (
                    total_padding - left_pad)

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        return self.conv(padded)

class StegaStampEncoder(nn.Module):
    """The image steganography encoder to implant the backdoor trigger.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size=20, height=32, width=32, in_channel=3):
        super(StegaStampEncoder, self).__init__()
        self.height, self.width, self.in_channel = height, width, in_channel

        self.secret_dense = nn.Sequential(nn.Linear(in_features=secret_size, out_features=height * width * in_channel), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential(Conv2dSame(in_channels=in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=256, kernel_size=3, stride=2), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=2), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(Conv2dSame(in_channels=256, out_channels=128, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=2), nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(Conv2dSame(in_channels=128, out_channels=64, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv8 = nn.Sequential(Conv2dSame(in_channels=64, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        # merge two branch feature like U-Net architecture
        self.up9 = nn.Sequential(Conv2dSame(in_channels=32, out_channels=32, kernel_size=2), nn.ReLU(inplace=True))
        self.conv9 = nn.Sequential(Conv2dSame(in_channels=64+in_channel*2, out_channels=32, kernel_size=3), nn.ReLU(inplace=True))

        self.residual = nn.Sequential(Conv2dSame(in_channels=32, out_channels=in_channel, kernel_size=1))

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = secret.reshape((-1, self.in_channel, self.height, self.width))
        inputs = torch.cat([secret, image], axis=1)

        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        up6 = self.up6(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv5))
        merge6 = torch.cat([conv4,up6], axis=1)
        conv6 = self.conv6(merge6)

        up7 = self.up7(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv6))
        merge7 = torch.cat([conv3,up7], axis=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv7))
        merge8 = torch.cat([conv2,up8], axis=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(nn.Upsample(scale_factor=(2, 2), mode='nearest')(conv8))
        merge9 = torch.cat([conv1,up9,inputs], axis=1)

        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)

        return residual


class StegaStampDecoder(nn.Module):
    """The image steganography decoder to assist the training of the image steganography encoder.

    We implement it based on the official tensorflow version:

    https://github.com/tancik/StegaStamp

    Args:
        secret_size (int): Size of the steganography secret.
        height (int): Height of the input image.
        width (int): Width of the input image.
        in_channel (int): Channel of the input image.
    """
    def __init__(self, secret_size, height, width, in_channel):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.in_channel = in_channel

        # Spatial transformer
        self.stn_params_former = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.stn_params_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2)*(width//2//2//2), out_features=128), nn.ReLU(inplace=True)
        )

        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = torch.FloatTensor(initial.astype('float32').flatten())

        self.W_fc1 = nn.Parameter(torch.zeros([128, 6]))
        self.b_fc1 = nn.Parameter(initial)

        self.decoder = nn.Sequential(
            Conv2dSame(in_channels=in_channel, out_channels=32, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
            Conv2dSame(in_channels=128, out_channels=128, kernel_size=3, stride=2), nn.ReLU(inplace=True),
        )

        self.decoder_later = nn.Sequential(
            nn.Linear(in_features=128*(height//2//2//2//2//2)*(width//2//2//2//2//2), out_features=512), nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=secret_size)
        )

    def forward(self, image):
        image = image - .5
        stn_params = self.stn_params_former(image)
        stn_params = stn_params.view(stn_params.size(0), -1)
        stn_params = self.stn_params_later(stn_params)

        x = torch.mm(stn_params, self.W_fc1) + self.b_fc1
        x = x.view(-1, 2, 3) # change it to the 2x3 matrix
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self.in_channel, self.height, self.width)))
        transformed_image = F.grid_sample(image, affine_grid_points)

        secret = self.decoder(transformed_image)
        secret = secret.view(secret.size(0), -1)
        secret = self.decoder_later(secret)
        return secret