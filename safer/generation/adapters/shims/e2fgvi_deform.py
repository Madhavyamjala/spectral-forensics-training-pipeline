"""
Deformable convolution v2 for E2FGVI, backed by torchvision instead of mmcv's compiled ops.

E2FGVI's feature propagation imports `ModulatedDeformConv2d` and `modulated_deform_conv2d`
from `mmcv.ops` - the half of mmcv that ships compiled CUDA kernels. mmcv 1.x lite carries
`mmcv.cnn` and `mmcv.runner` but no `mmcv._ext`, and building mmcv-full needs nvcc and hours.

torchvision already ships the operator. `deform_conv2d` with a `mask` argument *is* DCNv2,
with the same offset and mask layouts (`deform_groups * 2 * kh * kw` and `deform_groups *
kh * kw` channels) and the same weight shape `(out, in // groups, kh, kw)` - so E2FGVI's
released checkpoint loads into it unchanged. `groups` is read from the weight shape and the
offset groups from the offset channels, which is why the two trailing arguments are accepted
for signature compatibility and not forwarded.
"""

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


def _pair(value):
    """Upstream passes ints or pairs interchangeably; torchvision wants pairs."""
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)


def modulated_deform_conv2d(x, offset, mask, weight, bias=None, stride=1, padding=0,
                            dilation=1, groups=1, deform_groups=1):
    """mmcv.ops.modulated_deform_conv2d, on torchvision's kernel."""
    return deform_conv2d(x, offset, weight, bias, stride=_pair(stride),
                         padding=_pair(padding), dilation=_pair(dilation), mask=mask)


class ModulatedDeformConv2d(nn.Module):
    """The attributes E2FGVI's SecondOrderDeformableAlignment reads off its base class."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deform_groups=1, bias=True):
        """Match mmcv's constructor, including the parameter shapes the checkpoint expects."""
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = (0, 0)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.init_weights()

    def init_weights(self):
        """mmcv seeds the kernel this way; the checkpoint overwrites it either way."""
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, offset, mask):
        """Apply the modulated deformable convolution."""
        return modulated_deform_conv2d(x, offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups,
                                       self.deform_groups)
