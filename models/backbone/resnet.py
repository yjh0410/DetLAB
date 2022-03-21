# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

try:
    from .vision import model_resnet
except:
    from vision import model_resnet


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
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
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, num_channels: int):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {'layer2': 'layer2', 'layer3': 'layer3', 'layer4': 'layer4'}        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, x):
        xs = self.body(x)
        fmp_list = dict()
        for name, fmp in xs.items():
            fmp_list[name] = fmp

        return fmp_list


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, 
                 name: str,
                 pretrained: bool,
                 dilation: bool,
                 norm_type: str):
        if norm_type == 'BN':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'FrozeBN':
            norm_layer = FrozenBatchNorm2d
        backbone = getattr(model_resnet, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=norm_layer)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels)


def build_resnet(model_name='resnet18', pretrained=False, norm_type='BN'):
    if model_name in ['resnet18', 'resnet34']:
        backbone = Backbone(model_name, 
                            pretrained, 
                            dilation=False,
                            norm_type=norm_type)
        feats = [128, 256, 512] # [C3, C4, C5]
    
    elif model_name in ['resnet50', 'resnet101']:
        backbone = Backbone(model_name, 
                            pretrained, 
                            dilation=False,
                            norm_type=norm_type)
        feats = [512, 1024, 2048] # [C3, C4, C5]

    elif model_name in ['resnet50-d', 'resnet101-d']:
        backbone = Backbone(model_name[:-2], 
                            pretrained, 
                            dilation=True,
                            norm_type=norm_type)
        feats = [512, 1024, 2048] # [C3, C4, C5]

    return backbone, feats


if __name__ == '__main__':
    model, feat_dim = build_resnet(model_name='resnet18', pretrained=True)
    print(feat_dim)

    x = torch.randn(2, 3, 800, 800)
    outputs = model(x)
    # print(outputs['layer2'].shape)
    for k in outputs.keys():
        print(k)
        f = outputs[k]
        print(f.shape)
