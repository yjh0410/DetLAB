from turtle import forward
from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicFPN(nn.Module):
    def __init__(self, 
                 in_dims=[2048, 1024, 512],
                 out_dim=256,
                 ):
        super().__init__()
        # latter layers
        self.input_proj = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        
        for in_dim in in_dims:
            self.input_proj.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
            self.smooth_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1))


    def forward(self, feats):
        """
            feats: (List of Tensor)
        """
        outputs = []
        for i, x in enumerate(feats):
            if i == 0:
                x = self.smooth_layers[i](self.input_proj[i](x))
                outputs.append(x)
            else:
                x1 = self.input_proj[i](x)
                x2 = outputs[i - 1]
                x2_up = F.interpolate(x2, size=x1.shape[2:])
                y = self.smooth_layers[i](x1 + x2_up)
                outputs.append(y)

        return outputs
        