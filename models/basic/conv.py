import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwishImplementation(torch.autograd.Function):
    """
    Swish activation function memory-efficient implementation.

    This implementation explicitly processes the gradient, it keeps a copy of the input tensor,
    and uses it to calculate the gradient during the back-propagation phase.
    """
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    """
    Implement the Swish activation function.
    See: https://arxiv.org/abs/1710.05941 for more details.
    """
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_conv2d(c1, c2, k, p, s, d, g, padding_mode='ZERO', bias=False):
    if padding_mode == 'ZERO':
        conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)
    elif padding_mode == 'SAME':
        conv = Conv2dSamePadding(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv


def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type == 'swish':
        return Swish()
    elif act_type == 'swish-me':
        return MemoryEfficientSwish()


def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)


# Conv2d with "SAME" padding
class Conv2dSamePadding(nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support "SAME" padding mode and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """

        # parse padding mode
        self.padding_method = kwargs.pop("padding", None)
        if self.padding_method is None:
            if len(args) >= 5:
                self.padding_method = args[4]
            else:
                self.padding_method = 0  # default padding number

        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                # If the padding mode is `SAME`, it will be manually padded
                super().__init__(*args, **kwargs, padding=0)
                # stride
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                # kernel size
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
                # dilation
                if isinstance(self.dilation, int):
                    self.dilation = [self.dilation] * 2
                elif len(self.dilation) == 1:
                    self.dilation = [self.dilation[0]] * 2
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size
                dilation_h, dilation_w = self.dilation

                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)

                padding_needed_h = max(
                    0, (output_h - 1) * stride_h + (kernel_size_h - 1) * dilation_h + 1 - input_h
                )
                padding_needed_w = max(
                    0, (output_w - 1) * stride_w + (kernel_size_w - 1) * dilation_w + 1 - input_w
                )

                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top

                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))

        x = super().forward(x)

        return x


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # in channels
                 c2,                   # out channels 
                 k=1,                  # kernel size 
                 p=0,                  # padding
                 s=1,                  # padding
                 d=1,                  # dilation
                 act_type='',          # activation
                 norm_type='',         # normalization
                 padding_mode='ZERO',  # padding mode: "ZERO" or "SAME"
                 depthwise=False):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True
        if depthwise:
            assert c1 == c2, "In depthwise conv, the in_dim (c1) should be equal to out_dim (c2)."
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=c1, padding_mode=padding_mode, bias=add_bias))
            # depthwise conv
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            # pointwise conv
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, padding_mode=padding_mode, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)


# MaxPool2d with "SAME" padding
class MaxPool2dSamePadding(torch.nn.MaxPool2d):
    """
    A wrapper around :class:`torch.nn.MaxPool2d` to support "SAME" padding mode and more features.

    See: https://github.com/pytorch/pytorch/issues/3867
    """

    def __init__(self, *args, **kwargs):
        # parse padding mode
        self.padding_method = kwargs.pop("padding", None)
        if self.padding_method is None:
            if len(args) >= 3:
                self.padding_method = args[2]
            else:
                self.padding_method = 0  # default padding number

        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                # If the padding mode is `SAME`, it will be manually padded
                super().__init__(*args, **kwargs, padding=0)
                # stride
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                # kernel size
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size

                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)

                padding_needed_h = max(
                    0, (output_h - 1) * stride_h + (kernel_size_h - 1) + 1 - input_h
                )
                padding_needed_w = max(
                    0, (output_w - 1) * stride_w + (kernel_size_w - 1) + 1 - input_w
                )

                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top

                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))

        x = super().forward(x)
        return x


class ConvBlocks(nn.Module):
    def __init__(self, in_dim, out_dim, norm_type='BN', act_type='lrelu'):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            Conv(in_dim, in_dim//2, k=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim//2, in_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim, in_dim//2, k=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim//2, in_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        return self.conv_blocks(x)
