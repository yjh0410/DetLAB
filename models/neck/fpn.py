import torch
import torch.nn as nn
import torch.nn.functional as F


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
        assert from_c5 == p6_feat

        # latter layers
        self.input_proj = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        
        for in_dim in in_dims[::-1]:
            self.input_proj.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
            self.smooth_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1))

        # P6/P7
        if p6_feat:
            if from_c5:
                self.p6_conv = nn.Conv2d(in_dims[-1], out_dim, kernel_size=3, stride=2, padding=1)
            else: # from p5
                self.p6_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1)
        if p7_feat:
            self.p7_conv = nn.Sequential(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, feats):
        """
            feats: (List of Tensor) [C3, C4, C5], C_i ∈ R^(B x C_i x H_i x W_i)
        """
        outputs = []
        # [C3, C4, C5] -> [C5, C4, C3]
        feats = feats[::-1]
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

        # [P5, p4, P3] -> [P3, P4, P5]
        outputs = outputs[::-1]

        if self.p6_feat:
            if self.from_c5:
                P_6 = self.p6_conv(feats[0])
            else:
                P_6 = self.p6_conv(outputs[-1])
            # [P3, P4, P5] -> [P3, P4, P5, P6]
            outputs.append(P_6)

            if self.p7_feat:
                P_7 = self.p7_conv(outputs[-1])
                # [P3, P4, P5, P6] -> [P3, P4, P5, P6, P7]
                outputs.append(P_7)

        # [P3, P4, P5] or [P3, P4, P5, P6, P7]
        return outputs


class BiFPN(nn.Module):
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


class PaFPN(nn.Module):
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


class DynamicFPN(nn.Module):
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

