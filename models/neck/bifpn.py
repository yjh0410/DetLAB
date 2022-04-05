import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv, MemoryEfficientSwish, Swish, MaxPool2dSamePadding


class BiFPNLayer(nn.Module):
    """
    This module implements one layer of BiFPN, and BiFPN can be obtained
    by stacking this module multiple times.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, 
                 in_dims, 
                 out_dim,
                 fuse_type="fast", 
                 norm_type="BN", 
                 memory_efficient=True,
                 p6_feat=False,
                 p7_feat=False):
        """
        in_dims (list): the number of input tensor channels per level.
        out_dim (int): the number of output tensor channels.
        fuse_type (str): now only support three weighted fusion approaches:

            * fast:    Output = sum(Input_i * w_i / sum(w_j))
            * sotfmax: Output = sum(Input_i * e ^ w_i / sum(e ^ w_j))
            * sum:     Output = sum(Input_i) / len(Input_i)

        norm (str): the normalization to use.
        memory_efficient (bool): use `MemoryEfficientSwish` or `Swish` as activation function.
        """
        super(BiFPNLayer, self).__init__()
        assert fuse_type in ("fast", "softmax", "sum"), f"Unknown fuse method: {fuse_type}." \
            " Please select in [fast, sotfmax, sum]."

        self.fuse_type = fuse_type
        self.p6_feat = p6_feat
        self.p7_feat = p7_feat
        self.num_out_feats = len(in_dims)
        self.act = MemoryEfficientSwish() if memory_efficient else Swish()
        self.down_sampling = MaxPool2dSamePadding(kernel_size=3, stride=2, padding="SAME")

        # top down
        self.top_down_combine_weights = nn.ParameterList()
        self.top_down_combine_convs = nn.ModuleList()
        for _ in range(self.num_out_feats - 1):
            # combine weight
            if fuse_type == "fast" or fuse_type == "softmax":
                weights_i = nn.Parameter(
                    torch.ones(2, dtype=torch.float32), requires_grad=True,
                )
            elif fuse_type == "sum":
                weights_i = nn.Parameter(
                    torch.ones(2, dtype=torch.float32), requires_grad=False,
                )
            else:
                raise ValueError("Unknown fuse method: {}".format(self.fuse_type))
            self.top_down_combine_weights.append(weights_i)
            combine_conv = Conv(out_dim, out_dim, k=3, p=1, s=1, padding_mode="SAME", norm_type=norm_type, act_type=None, depthwise=True)
            self.top_down_combine_convs.append(combine_conv)
        # bottom up
        self.bottom_up_combine_weights = nn.ParameterList()
        self.bottom_up_combine_convs = nn.ModuleList()
        for j in range(self.num_out_feats - 1):
            if j == self.num_out_feats - 2:
                # combine weight
                if fuse_type == "fast" or fuse_type == "softmax":
                    weights_i = nn.Parameter(
                        torch.ones(2, dtype=torch.float32), requires_grad=True,
                    )
                elif fuse_type == "sum":
                    weights_i = nn.Parameter(
                        torch.ones(2, dtype=torch.float32), requires_grad=False,
                    )
                else:
                    raise ValueError("Unknown fuse method: {}".format(self.fuse_type))
                self.bottom_up_combine_weights.append(weights_i)
                combine_conv = Conv(out_dim, out_dim, k=3, p=1, s=1, padding_mode="SAME", norm_type=norm_type, act_type=None, depthwise=True)
                self.bottom_up_combine_convs.append(combine_conv)
            else:
                # combine weight
                if fuse_type == "fast" or fuse_type == "softmax":
                    weights_i = nn.Parameter(
                        torch.ones(3, dtype=torch.float32), requires_grad=True,
                    )
                elif fuse_type == "sum":
                    weights_i = nn.Parameter(
                        torch.ones(3, dtype=torch.float32), requires_grad=False,
                    )
                else:
                    raise ValueError("Unknown fuse method: {}".format(self.fuse_type))
                self.bottom_up_combine_weights.append(weights_i)
                combine_conv = Conv(out_dim, out_dim, k=3, p=1, s=1, padding_mode="SAME", norm_type=norm_type, act_type=None)
                self.bottom_up_combine_convs.append(combine_conv)


    def forward(self, in_feats):
        # top down fpn
        inter_feats = []
        # [P3, P4, P5, P6, P7] -> [P7, P6, P5, P4, P3]
        in_feats = in_feats[::-1]
        top_level_feat = in_feats[0]
        prev_feat = top_level_feat
        inter_feats.append(prev_feat)

        for feat, weights_i, smooth in zip(in_feats[1:], self.top_down_combine_weights, self.top_down_combine_convs):
            # edge weights
            if self.fuse_type == "fast":
                weights_i = F.relu(weights_i)
            elif self.fuse_type == "softmax":
                weights_i = weights_i.softmax(dim=0)
            elif self.fuse_type == "sum":
                weights_i = weights_i
            weights = torch.div(weights_i, weights_i.sum() + 1e-4)
            top_down_feat = F.interpolate(prev_feat, size=feat.shape[2:], mode='nearest')
            prev_feat = weights[0] * feat + weights[1] * top_down_feat
            inter_feats.insert(0, smooth(self.act(prev_feat)))
        
        # Finally, inter_feats contains [P3_inter, P4_inter, P5_inter, P6_inter, P7_inter]
        # [P7, P6, P5, P4, P3] -> [P3, P4, P5, P6, P7]
        in_feats = in_feats[::-1]
        # bottom up fpn
        out_feats = []
        bottom_level_feat = inter_feats[0]
        prev_feat = bottom_level_feat
        out_feats.append(prev_feat)
        for idx, (orig_feat, inter_feat, weights_i, smooth) in enumerate(zip(in_feats[1:], 
                                                                           inter_feats[1:],
                                                                           self.bottom_up_combine_weights,
                                                                           self.bottom_up_combine_convs)):
            # edge weights
            if self.fuse_type == "fast":
                weights_i = F.relu(weights_i)
            elif self.fuse_type == "softmax":
                weights_i = weights_i.softmax(dim=0)
            elif self.fuse_type == "sum":
                weights_i = weights_i
            weights = torch.div(weights_i, weights_i.sum() + 1e-4)
            bottom_up_feat = self.down_sampling(prev_feat)
            if idx == len(in_feats[1:]) - 1:
                prev_feat = weights[0] * inter_feat + weights[1] * bottom_up_feat
                out_feats.append(smooth(self.act(prev_feat)))
            else:
                prev_feat = weights[0] * orig_feat + weights[1] * inter_feat + weights[2] * bottom_up_feat
                out_feats.append(smooth(self.act(prev_feat)))

        return out_feats
        

class BiFPN(nn.Module):
    """
    This module implements the BIFPN module in EfficientDet.
    See: https://arxiv.org/pdf/1911.09070.pdf for more details.
    """

    def __init__(self, 
                 in_dims, # [..., C3, C4, C5, ...]
                 out_dim, 
                 num_bifpn_layers=1,
                 fuse_type="fast", 
                 norm_type="BN", 
                 bn_momentum=0.01, 
                 bn_eps=1e-3,
                 memory_efficient=True,
                 p6_feat=False,
                 p7_feat=False):
        super(BiFPN, self).__init__()
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.p6_feat = p6_feat
        self.p7_feat = p7_feat
        self.num_out_feats = len(in_dims)

        # input proj layers
        self.input_projs = nn.ModuleList()
        for in_dim in in_dims:
            self.input_projs.append(Conv(in_dim, out_dim, k=1, p=0, s=1, norm_type=norm_type, act_type=None))

        if p6_feat:
            self.p6_layer = nn.Sequential(
                Conv(in_dims[-1], out_dim, k=1, p=0, s=1, norm_type=norm_type, act_type=None),
                MaxPool2dSamePadding(kernel_size=3, stride=2, padding="SAME")
            )
            self.num_out_feats += 1
        if p7_feat:
            self.p7_layer = MaxPool2dSamePadding(kernel_size=3, stride=2, padding="SAME")
            self.num_out_feats += 1

        # build bifpn layers
        self.bifpn_layers = nn.ModuleList()
        for _ in range(num_bifpn_layers):
            bifpn_layer_in_channels = [out_dim] * self.num_out_feats
            bifpn_layer = BiFPNLayer(bifpn_layer_in_channels, out_dim, fuse_type, norm_type, memory_efficient)
            self.bifpn_layers.append(bifpn_layer)

        self._init_weights()


    def _init_weights(self):
        """
        Weight initialization as per Tensorflow official implementations.
        See: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/init_ops.py
             #L437
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_in = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                stddev = math.sqrt(1. / max(1., fan_in))
                m.weight.data.normal_(0, stddev)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if self.bn_momentum is not None and self.bn_eps is not None:
                    m.momentum = self.bn_momentum
                    m.eps = self.bn_eps
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, feats):
        # input projection
        results = []
        for feat, layer in zip(feats, self.input_projs):
            results.append(layer(feat))
        if self.p6_feat:
            p6_feat = self.p6_layer(feats[-1])
            results.append(p6_feat)
        if self.p7_feat:
            p7_feat = self.p7_layer(results[-1])
            results.append(p7_feat)

        # build top-down and bottom-up path with stack
        for bifpn_layer in self.bifpn_layers:
            results = bifpn_layer(results)

        return results


if __name__ == '__main__':
    c3 = torch.randn(2, 32, 256, 256)
    c4 = torch.randn(2, 64, 128, 128)
    c5 = torch.randn(2, 128, 64, 64)
    bifpn = BiFPN(in_dims=[32, 64, 128],
                  out_dim=32, 
                  num_bifpn_layers=1,
                  fuse_type="softmax", 
                  norm_type="BN", 
                  bn_momentum=0.01, 
                  bn_eps=1e-3,
                  memory_efficient=True,
                  p6_feat=True,
                  p7_feat=True)

    results = bifpn([c3, c4, c5])
    for y in results:
        print(y.shape)
