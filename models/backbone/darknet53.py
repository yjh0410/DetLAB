import torch
import torch.nn as nn
import os


model_urls = {
    "darknet53": "https://github.com/yjh0410/DetLAB/releases/download/object-detection-benchmark-backbone/darknet53.pth",
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


class ResBlock(nn.Module):
    def __init__(self, ch, nblocks=1, norm_type='BN'):
        super().__init__()
        self.module_list = nn.ModuleList()
        for _ in range(nblocks):
            resblock_one = nn.Sequential(
                Conv_BN_LeakyReLU(ch, ch//2, ksize=1, norm_type=norm_type),
                Conv_BN_LeakyReLU(ch//2, ch, ksize=3, padding=1, norm_type=norm_type)
            )
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            x = module(x) + x
        return x


class DarkNet_53(nn.Module):
    """
    DarkNet-53.
    """
    def __init__(self, norm_type='BN'):
        super(DarkNet_53, self).__init__()
        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, ksize=3, padding=1, norm_type=norm_type),
            Conv_BN_LeakyReLU(32, 64, ksize=3, padding=1, stride=2, norm_type=norm_type),
            ResBlock(64, nblocks=1, norm_type=norm_type)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, ksize=3, padding=1, stride=2, norm_type=norm_type),
            ResBlock(128, nblocks=2, norm_type=norm_type)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, ksize=3, padding=1, stride=2, norm_type=norm_type),
            ResBlock(256, nblocks=8, norm_type=norm_type)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, ksize=3, padding=1, stride=2, norm_type=norm_type),
            ResBlock(512, nblocks=8, norm_type=norm_type)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, ksize=3, padding=1, stride=2, norm_type=norm_type),
            ResBlock(1024, nblocks=4, norm_type=norm_type)
        )

        self.freeze()


    def freeze(self):
        print('freeze parameters of one & two layers ...')
        for m in self.layer_1.parameters():
            m.requires_grad = False
        for m in self.layer_2.parameters():
            m.requires_grad = False
        

    def forward(self, x):
        outputs = dict()
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs["layer2"] = c3
        outputs["layer3"] = c4
        outputs["layer4"] = c5

        return outputs


def darknet53(pretrained=False, norm_type='BN'):
    """Constructs a darknet-53 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_53(norm_type=norm_type)
    feats = [256, 512, 1024] # C3, C4, C5

    if pretrained:
        print('Loading pretrained darknet53 ...')
        url = model_urls['darknet53']
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
    m, f = darknet53(pretrained=True, norm_type='FrozeBN')
    m.eval()
    out = m(x)

    for k in out.keys():
        y = out[k]
        print(y.size())
