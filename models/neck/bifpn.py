import torch
import torch.nn as nn
import torch.nn.functional as F

from ..basic.conv import Conv


class BiFPN(nn.Module):
    def __init__(self,
                 in_dims = [512, 1024, 2048],
                 out_dim=256,
                 from_c5=False,
                 p6_feat=False,
                 p7_feat=False):
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
