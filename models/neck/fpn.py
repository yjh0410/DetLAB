import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weight_init
from ..basic.conv import Conv, ConvBlocks
from ..basic.bottleneck_csp import BottleneckCSP
from .extra_module import SPPBlock, SPPBlockCSP, build_neck


class BasicFPN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 2048],
                 out_dim=256,
                 from_c5=False,
                 p6_feat=False,
                 p7_feat=False
                 ):
        super().__init__()
        self.from_c5 = from_c5
        self.p6_feat = p6_feat
        self.p7_feat = p7_feat

        # latter layers
        self.input_projs = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        
        for in_dim in in_dims[::-1]:
            self.input_projs.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
            self.smooth_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1))

        # P6/P7
        if p6_feat:
            if from_c5:
                self.p6_conv = nn.Conv2d(in_dims[-1], out_dim, kernel_size=3, stride=2, padding=1)
            else: # from p5
                self.p6_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        if p7_feat:
            self.p7_conv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
            )

        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)


    def forward(self, feats):
        """
            feats: (List of Tensor) [C3, C4, C5], C_i ∈ R^(B x C_i x H_i x W_i)
        """
        outputs = []
        # [C3, C4, C5] -> [C5, C4, C3]
        feats = feats[::-1]
        top_level_feat = feats[0]
        prev_feat = self.input_projs[0](top_level_feat)
        outputs.append(self.smooth_layers[0](prev_feat))

        for feat, input_proj, smooth_layer in zip(feats[1:], self.input_projs[1:], self.smooth_layers[1:]):
            feat = input_proj(feat)
            top_down_feat = F.interpolate(prev_feat, size=feat.shape[2:], mode='nearest')
            prev_feat = feat + top_down_feat
            outputs.insert(0, smooth_layer(prev_feat))

        if self.p6_feat:
            if self.from_c5:
                p6_feat = self.p6_conv(feats[0])
            else:
                p6_feat = self.p6_conv(outputs[-1])
            # [P3, P4, P5] -> [P3, P4, P5, P6]
            outputs.append(p6_feat)

            if self.p7_feat:
                p7_feat = self.p7_conv(p6_feat)
                # [P3, P4, P5, P6] -> [P3, P4, P5, P6, P7]
                outputs.append(p7_feat)

        # [P3, P4, P5] or [P3, P4, P5, P6, P7]
        return outputs


class PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 2048], # [..., C3, C4, C5, ...]
                 out_dim=256,
                 norm_type='',
                 p6_feat=False,
                 p7_feat=False
                 ):
        super().__init__()
        self.p6_feat = p6_feat
        self.p7_feat = p7_feat
        self.num_fpn_feats = len(in_dims)

        # input projection layers
        self.input_projs = nn.ModuleList()
        
        for in_dim in in_dims:
            self.input_projs.append(Conv(in_dim, out_dim, k=1, p=0, s=1, act_type=None, norm_type=norm_type))

        # P6/P7
        if p6_feat:
            self.p6_layer = Conv(in_dims[-1], out_dim, k=3, p=1, s=2, act_type=None, norm_type=norm_type)
            self.num_fpn_feats += 1

        if p7_feat:
            self.p7_layer = Conv(out_dim, out_dim, k=3, p=1, s=2, act_type=None, norm_type=norm_type)
            self.num_fpn_feats += 1

        # top_down_smooth_layers
        self.top_down_smooth_layers = nn.ModuleList()
        for i in range(self.num_fpn_feats):
            self.top_down_smooth_layers.append(Conv(out_dim, 
                                                    out_dim, 
                                                    k=3, 
                                                    p=1, 
                                                    s=1, 
                                                    act_type=None, 
                                                    norm_type=norm_type))

        # bottom_up_smooth_layers
        self.bottom_up_smooth_layers = nn.ModuleList()
        self.bottom_up_downsample_layers = nn.ModuleList()
        for i in range(self.num_fpn_feats):
            self.bottom_up_smooth_layers.append(Conv(out_dim, 
                                                     out_dim, 
                                                     k=3, 
                                                     p=1, 
                                                     s=1, 
                                                     act_type=None, 
                                                     norm_type=norm_type))
            if i > 0:
                self.bottom_up_downsample_layers.append(Conv(out_dim, 
                                                             out_dim, 
                                                             k=3, 
                                                             p=1, 
                                                             s=2, 
                                                             act_type=None, 
                                                             norm_type=norm_type))
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)


    def forward(self, feats):
        """
            feats: (List of Tensor) [C3, C4, C5], C_i ∈ R^(B x C_i x H_i x W_i)
        """
        in_feats = []
        for feat, layer in zip(feats, self.input_projs):
            in_feats.append(layer(feat))

        # top down fpn
        inter_feats = []
        in_feats = in_feats[::-1]    # [..., C3, C4, C5, ...] -> [..., C5, C4, C3, ...]
        top_level_feat = in_feats[0]
        prev_feat = top_level_feat
        inter_feats.append(self.top_down_smooth_layers[0](prev_feat))

        for feat, smooth in zip(in_feats[1:], self.top_down_smooth_layers[1:]):
            # upsample
            top_down_feat = F.interpolate(prev_feat, size=feat.shape[2:], mode='nearest')
            # sum
            prev_feat = feat + top_down_feat
            inter_feats.insert(0, smooth(prev_feat))

        # Finally, inter_feats contains [P3_inter, P4_inter, P5_inter, P6_inter, P7_inter]
        # bottom up fpn
        out_feats = []
        bottom_level_feat = inter_feats[0]
        prev_feat = bottom_level_feat
        out_feats.append(self.bottom_up_smooth_layers[0](prev_feat))
        for inter_feat, smooth, downsample in zip(inter_feats[1:], 
                                                  self.bottom_up_smooth_layers[1:], 
                                                  self.bottom_up_downsample_layers):
            # downsample
            bottom_up_feat = downsample(prev_feat)
            # sum
            prev_feat = inter_feat + bottom_up_feat
            out_feats.append(smooth(prev_feat))

        return out_feats


class YoloFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 out_dim=None,
                 norm_type='BN',
                 act_type='lrelu',
                 spp=False):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
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
        self.head_conv_2 = Conv(c4//2, c4, k=3, p=1, norm_type=norm_type, act_type=act_type)

        # P8/32-large
        self.head_convblock_2 = ConvBlocks(c3 + c3//2, c3//2, norm_type=norm_type, act_type=act_type)
        self.head_conv_3 = Conv(c3//2, c3, k=3, p=1, norm_type=norm_type, act_type=act_type)

        # output proj layers
        if out_dim is not None:
            self.out_layers = nn.ModuleList([Conv(in_dim, out_dim, k=1, norm_type=norm_type, act_type=act_type) for in_dim in in_dims])


        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.head_convblock_0(c5)
        p5_up = F.interpolate(self.top_down_conv_0(p5), size=c4.shape[2:], mode='nearest')
        p5 = self.head_conv_1(p5)

        # p4/16
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = F.interpolate(self.top_down_conv_1(p4), size=c3.shape[2:], mode='nearest')
        p4 = self.head_conv_2(p4)

        # P3/8
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_3(p3)

        out_feats = [p3, p4, p5]
        # output proj layers
        if self.out_dim is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


# YoloPaFPN
class YoloPaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[256, 512, 1024],
                 out_dim=None, 
                 depth=1.0, 
                 depthwise=False,
                 norm_type='BN',
                 act_type='lrelu',
                 spp=False):
        super(YoloPaFPN, self).__init__()
        c3, c4, c5 = in_dims
        nblocks = int(3 * depth)
        if spp:
            self.head_csp_0 = SPPBlockCSP(c5, c5, e=0.5, norm_type=norm_type, act_type=act_type)
        else:
            self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act_type=act_type, norm_type=norm_type)
        self.top_down_conv_0 = Conv(c5, c5//2, k=1, act_type=act_type, norm_type=norm_type)  # 10
        self.head_csp_1 = BottleneckCSP(c4 + c5//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act_type=act_type, norm_type=norm_type)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1, act_type=act_type, norm_type=norm_type)  # 14
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c3, n=nblocks, shortcut=False, depthwise=depthwise, act_type=act_type, norm_type=norm_type)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2, depthwise=depthwise, act_type=act_type, norm_type=norm_type)
        self.head_csp_3 = BottleneckCSP(c3 + c4//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act_type=act_type, norm_type=norm_type)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2, depthwise=depthwise, act_type=act_type, norm_type=norm_type)
        self.head_csp_4 = BottleneckCSP(c4 + c5//2, c5, n=nblocks, shortcut=False, depthwise=depthwise, norm_type=norm_type, act_type=act_type)

        # output proj layers
        if out_dim is not None:
            self.out_layers = nn.ModuleList([Conv(in_dim, out_dim, k=1, norm_type=norm_type, act_type=act_type) for in_dim in in_dims])


    def forward(self, features):
        c3, c4, c5 = features

        c5 = self.head_csp_0(c5)
        c6 = self.top_down_conv_0(c5)
        c7 = F.interpolate(c6, scale_factor=2.0, mode='nearest')   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_1(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = F.interpolate(c10, scale_factor=2.0, mode='nearest')  # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_2(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_3(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_4(c18)  # to det

        out_feats = [c13, c16, c19] # [P3, P4, P5]
        # output proj layers
        if self.out_dim is not None:
            out_feats_proj = []
            for feat, layer in zip(out_feats, self.out_layers):
                out_feats_proj.append(layer(feat))
            return out_feats_proj

        return out_feats


def build_fpn(cfg, in_dims, out_dim, from_c5=False, p6_feat=False, p7_feat=False):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'basic_fpn':
        fpn_net = BasicFPN(in_dims=in_dims, 
                           out_dim=out_dim,
                           from_c5=from_c5, 
                           p6_feat=p6_feat,
                           p7_feat=p7_feat)

    elif model == 'pafpn':
        fpn_net = PaFPN(in_dims=in_dims,
                        out_dim=out_dim, 
                        norm_type=cfg['fpn_norm'], 
                        p6_feat=p6_feat,
                        p7_feat=p7_feat)

                            
    elif model == 'yolo_fpn':
        fpn_net = YoloFPN(in_dims=in_dims,
                          norm_type=cfg['fpn_norm'],
                          act_type=cfg['fpn_act'],
                          spp=cfg['use_spp'],
                          out_dim=cfg['head_dim'])

    elif model == 'yolo_pafpn':
        fpn_net = YoloPaFPN(in_dims=in_dims,
                            depth=cfg['depth'],
                            depthwise=cfg['depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'],
                            spp=cfg['use_spp'],
                            out_dim=cfg['head_dim'])

    return fpn_net
