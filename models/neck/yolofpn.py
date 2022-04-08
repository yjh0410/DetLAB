import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weight_init
from ..basic.conv import Conv
from .spp import SPPBlock


class ConvBlocks(nn.Module):
    def __init__(self, in_dim, norm_type='BN', act_type='lrelu'):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            Conv(in_dim, in_dim//2, k=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim//2, in_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim, in_dim//2, k=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim//2, in_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim, in_dim//2, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return self.conv_blocks(x)


class YoloFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 norm_type='BN',
                 act_type='lrelu',
                 spp=False):
        super().__init__()
        c3, c4, c5 = in_dims
        # head
        # P3/8-small
        if spp:
            self.head_convblock_0 = SPPBlock(c5, c5//2, norm_type=norm_type, act_type=act_type)
        else:
            self.head_convblock_0 = ConvBlocks(c5, c5//2, norm_type=norm_type, act_type=act_type)
        self.top_down_conv_0 = Conv(c5//2, c4//2, k=1, norm_type=norm_type, act_type=act_type)
        self.head_conv_1 = Conv(c5//2, c5, k=3, p=1, norm_type=norm_type, act_type=act_type)

        # P4/16-medium
        self.head_convblock_1 = ConvBlocks(c4 + c4//2, c4//2, norm_type=norm_type, act_type=act_type)
        self.top_down_conv_1 = Conv(c4//2, c3//2, k=1, norm_type=norm_type, act_type=act_type)
        self.head_conv_3 = Conv(c4//2, c4, k=3, p=1, norm_type=norm_type, act_type=act_type)

        # P8/32-large
        self.head_convblock_2 = ConvBlocks(c3 + c3//2, c3//2, norm_type=norm_type, act_type=act_type)
        self.head_conv_4 = Conv(c3//2, c3, k=3, p=1, norm_type=norm_type, act_type=act_type)


    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.head_convblock_0(c5)
        p5_up = F.interpolate(self.top_down_conv_0(p5), scale_factor=2.0, mode='nearest')
        p5 = self.head_conv_1(p5)

        # p4/16
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = F.interpolate(self.top_down_conv_1(p4), scale_factor=2.0, mode='nearest')
        p4 = self.head_conv_3(p4)

        # P3/8
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_4(p3)

        return [p3, p4, p5]

