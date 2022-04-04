import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv, MemoryEfficientSwish, Swish, MaxPool2dSamePadding, Upsample


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

        self.in_dims = in_dims
        self.fuse_type = fuse_type
        self.p6_feat = p6_feat
        self.p7_feat = p7_feat
        self.num_out_feats = len(in_dims)
        self.act = MemoryEfficientSwish() if memory_efficient else Swish()
        self.down_sampling = MaxPool2dSamePadding(kernel_size=3, stride=2, padding="SAME")
        self.up_sampling = Upsample(scale_factor=2, mode='nearest')

        # input proj layers
        self.input_projs = nn.ModuleList()
        for in_dim in in_dims:
            self.input_projs.append(Conv(in_dim, out_dim, k=1, p=0, s=1, norm_type=norm_type, act_type=None))

        if p6_feat:
            self.p6_layer = nn.Sequential(
                Conv(in_dim[-1], out_dim, k=1, p=0, s=1, norm_type=norm_type, act_type=None),
                MaxPool2dSamePadding(kernel_size=3, stride=2, padding="SAME")
            )
            self.num_out_feats += 1
        if p7_feat:
            self.p7_layer = MaxPool2dSamePadding(kernel_size=3, stride=2, padding="SAME")
            self.num_out_feats += 1

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
            combine_conv = Conv(out_dim, out_dim, k=3, p=1, s=1, padding_mode="SAME", norm_type=norm_type, act_type=None)
            self.top_down_combine_convs.append(combine_conv)
        # bottom up
        self.bottom_up_combine_weights = nn.ParameterList()
        self.bottom_up_combine_convs = nn.ModuleList()
        for _ in range(self.num_out_feats - 1):
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


    def forward(self, feats):
        # input projection
        in_feats = []
        for feat, layer in zip(feats, self.input_projs):
            in_feats.append(layer(feat))
        if self.p6_feat:
            p6_feat = self.p6_layer(in_feats[-1])
            in_feats.append(p6_feat)
        if self.p7_feat:
            p7_feat = self.p7_layer(in_feats[-1])
            in_feats.append(p7_feat)

        # top down fpn
        inter_feats = []
        # TO DO:
        
        # bottom up fpn
        out_feats = []
        # TO DO:
        
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
                 fuse_type="weighted_sum", 
                 norm_type="BN", 
                 bn_momentum=0.01, 
                 bn_eps=1e-3,
                 memory_efficient=True,
                 p6_feat=False,
                 p7_feat=False):
        """
        bottom_up (Backbone): module representing the bottom up subnetwork.
            Must be a subclass of :class:`Backbone`. The multi-scale feature
            maps generated by the bottom up network, and listed in `in_features`,
            are used to generate FPN levels.
        in_features (list[str]): names of the input feature maps coming
            from the backbone to which FPN is attached. For example, if the
            backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
            of these may be used; order must be from high to low resolution.
        out_dim (int): the number of channels in the output feature maps.
        num_bifpn_layers (str): the number of bifpn layer.
        fuse_type (str): weighted feature fuse type. see: `BiFPNLayer`
        top_block (nn.Module or None): if provided, an extra operation will
            be performed on the output of the last (smallest resolution)
            FPN output, and the result will extend the result list. The top_block
            further downsamples the feature map. It must have an attribute
            "num_levels", meaning the number of extra FPN levels added by
            this block, and "in_feature", which is a string representing
            its input feature (e.g., p5).
        norm (str): the normalization to use.
        bn_momentum (float): the `momentum` parameter of the norm module.
        bn_eps (float): the `eps` parameter of the norm module.
        """
        super(BiFPN, self).__init__()
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.p6_feat = p6_feat
        self.p7_feat = p7_feat
        self.num_out_feats = len(in_dims)

        # latter layers
        self.input_projs = nn.ModuleList()

        for in_dim in in_dims:
            self.input_projs.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))

        # P6/P7
        if p6_feat:
            self.p6_layer = nn.Sequential(
                Conv(in_dims[-1], out_dim, k=3, p=1, s=1, norm_type=norm_type, act_type=None),
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
        results = []
        for feat, layer in zip(feats, self.input_projs):
            in_feat = layer(feat)
            results.append(in_feat)

        # build top-down and bottom-up path with stack
        for bifpn_layer in self.bifpn_layers:
            results = bifpn_layer(results)

        return results


if __name__ == '__main__':
    c3 = torch.randn(2, 32, 64, 64)
    c4 = torch.randn(2, 64, 32, 32)
    c5 = torch.randn(2, 128, 16, 16)
    bifpn = BiFPN(in_dims=[32, 64, 128],
                  out_dim=32, 
                  num_bifpn_layers=1,
                  fuse_type="fast", 
                  norm_type="BN", 
                  bn_momentum=0.01, 
                  bn_eps=1e-3,
                  memory_efficient=True,
                  p6_feat=False,
                  p7_feat=False)

    results = bifpn([c3, c4, c5])
