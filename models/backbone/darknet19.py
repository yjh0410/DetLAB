import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys


model_urls = {
    "darknet19": "https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-backbone/darknet19.pth",
}


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, norm_type='BN'):
        super(Conv_BN_LeakyReLU, self).__init__()
        convs = []
        convs.append(nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation))
        if norm_type == 'BN':
            convs.append(nn.BatchNorm2d(out_channels))
        elif norm_type == 'FrozeBN':
            convs.append(FrozenBatchNorm2d(out_channels))
        convs.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self, norm_type='BN'):        
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, ksize=3, padding=1, norm_type=norm_type),
            nn.MaxPool2d((2, 2), 2),
        )

        # output : stride = 4, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, ksize=3, padding=1, norm_type=norm_type),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 128
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(128, 64, ksize=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(64, 128, ksize=3, padding=1, norm_type=norm_type),
            nn.MaxPool2d((2,2), 2)
        )

        # output : stride = 8, c = 256
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(256, 128, ksize=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(128, 256, ksize=3, padding=1, norm_type=norm_type),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(512, 256, ksize=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(256, 512, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(512, 256, ksize=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(256, 512, ksize=3, padding=1, norm_type=norm_type),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(1024, 512, ksize=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(512, 1024, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(1024, 512, ksize=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(512, 1024, ksize=3, padding=1, norm_type=norm_type)
        )

        self.freeze()


    def freeze(self):
        print('freeze parameters of one & two layers ...')
        for m in self.conv_1.parameters():
            m.requires_grad = False
        for m in self.conv_2.parameters():
            m.requires_grad = False
        for m in self.conv_3.parameters():
            m.requires_grad = False


    def forward(self, x):
        outputs = dict()
        c1 = self.conv_1(x)
        c2 = self.conv_2(c1)
        c3 = self.conv_3(c2)
        c3 = self.conv_4(c3)
        c4 = self.conv_5(self.maxpool_4(c3))
        c5 = self.conv_6(self.maxpool_5(c4))

        outputs["layer2"] = c3
        outputs["layer3"] = c4
        outputs["layer4"] = c5

        return outputs


def darknet19(pretrained=False, norm_type='BN'):
    """Constructs a darknet-19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_19(norm_type=norm_type)
    feats = [256, 512, 1024] # C3, C4, C5

    # load weight
    if pretrained:
        print('Loading pretrained darknet19 ...')
        url = model_urls['darknet19']
        checkpoint_state_dict = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)

        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict)

    return model, feats


if __name__ == '__main__':
    x = torch.ones(2, 3, 64, 64)
    m, f = darknet19(pretrained=True, norm_type='FrozeBN')
    m.eval()
    out = m(x)

    for k in out.keys():
        y = out[k]
        print(y.size())
