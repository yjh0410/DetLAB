import torch
import torch.nn as nn

from ..basic.conv import Conv
from utils import weight_init


# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], norm_type='BN', act_type='lrelu'):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        
        self.cv2 = Conv(c_*(len(kernel_sizes) + 1), c2, k=1, act_type=act_type, norm_type=norm_type)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cv2(x)

        return x


# SPP block
class SPPBlock(nn.Module):
    """
        Spatial Pyramid Pooling Block
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act_type='lrelu', norm_type='BN'):
        super(SPPBlock, self).__init__()
        self.m = nn.Sequential(
            Conv(c1, c1//2, k=1, act_type=act_type, norm_type=norm_type),
            Conv(c1//2, c1, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SPP(c1, c1//2, e=e, kernel_sizes=kernel_sizes, act_type=act_type, norm_type=norm_type),
            Conv(c1//2, c1, k=3, p=1, act_type=act_type, norm_type=norm_type),
            Conv(c1, c2, k=1, act_type=act_type, norm_type=norm_type)
        )

        
    def forward(self, x):
        x = self.m(x)

        return x


# SPP block with CSP module
class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act_type='lrelu', norm_type='BN'):
        super(SPPBlockCSP, self).__init__()
        self.cv1 = Conv(c1, c1//2, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(c1, c1//2, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(
            Conv(c1//2, c1//2, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SPP(c1//2, c1//2, e=e, kernel_sizes=kernel_sizes, act_type=act_type, norm_type=norm_type),
            Conv(c1//2, c1//2, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )
        self.cv3 = Conv(c1, c2, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


# Dilated Encoder
class Bottleneck(nn.Module):
    def __init__(self, 
                 in_dim, 
                 dilation=1, 
                 expand_ratio=0.25,
                 act_type='relu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.branch = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=dilation, d=dilation, act_type=act_type, norm_type=norm_type),
            Conv(inter_dim, in_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 expand_ratio=0.25, 
                 dilation_list=[2, 4, 6, 8],
                 act_type='relu',
                 norm_type='BN'):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_dim, out_dim, k=1, act_type=None, norm_type=norm_type),
            Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type=norm_type)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(in_dim=out_dim, 
                                       dilation=d, 
                                       expand_ratio=expand_ratio, 
                                       act_type=act_type,
                                       norm_type=norm_type))
        self.encoders = nn.Sequential(*encoders)

        self._init_weight()

    def _init_weight(self):
        for m in self.projector:
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)
                weight_init.c2_xavier_fill(m)
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.encoders.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'dilated_encoder':
        neck = DilatedEncoder(in_dim, 
                              out_dim, 
                              expand_ratio=cfg['expand_ratio'], 
                              dilation_list=cfg['dilation_list'],
                              act_type=cfg['act_type'])
    elif model == 'spp':
        neck = SPP(in_dim, 
                   out_dim, 
                   e=cfg['expand_ratio'], 
                   kernel_sizes=cfg['kernel_sizes'],
                   norm_type=cfg['neck_norm'],
                   act_type=cfg['neck_act'])

    elif model == 'spp_block':
        neck = SPPBlock(in_dim, 
                        out_dim, 
                        e=cfg['expand_ratio'], 
                        kernel_sizes=cfg['kernel_sizes'],
                        norm_type=cfg['neck_norm'],
                        act_type=cfg['neck_act'])


    return neck
