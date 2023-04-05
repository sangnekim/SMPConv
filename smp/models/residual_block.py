# torch
import torch
from timm.models.layers import trunc_normal_

# typing
from functools import partial
from typing import Tuple, Union
from ckconv.nn import CKConv, CausalConv1d
from torch.nn import Conv1d, Conv2d


class ResidualBlockBase(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ConvType: torch.nn.Module,
        NonlinearType: torch.nn.Module,
        NormType: torch.nn.Module,
        LinearType: torch.nn.Module,
        dropout: float,
        dim_linear: int,
        small_kernel_size: int
    ):
        """
        Instantiates the core elements of a residual block but does not implement the forward function.
        These elements are:
        (1) Two convolutional layers
        (2) Two normalization layers
        (3) A residual connection
        (4) A dropout layer
        """
        super().__init__()

        # Conv Layers
        self.cconv1 = ConvType(in_channels=in_channels, out_channels=out_channels)
        self.cconv2 = ConvType(in_channels=out_channels, out_channels=out_channels)

        # additional small kernel Conv layers
        if dim_linear == 1:
            self.cconv1s = CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=small_kernel_size, bias=False)
            self.cconv2s = CausalConv1d(in_channels=out_channels, out_channels=out_channels,
                                            kernel_size=small_kernel_size, bias=False)
        elif dim_linear == 2:
            self.cconv1s = Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=small_kernel_size, padding=small_kernel_size//2, bias=False)
            self.cconv2s = Conv2d(in_channels=out_channels, out_channels=out_channels,
                                    kernel_size=small_kernel_size, padding=small_kernel_size//2, bias=False)

        # Nonlinear layer
        self.nonlinear = NonlinearType()

        # Norm layers
        self.norm1 = NormType(out_channels)
        self.norm2 = NormType(out_channels)
        
        self.norm1s = NormType(out_channels)
        self.norm2s = NormType(out_channels)

        # Dropout
        self.dp = torch.nn.Dropout(dropout)

        # Shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(LinearType(in_channels, out_channels))
        self.shortcut = torch.nn.Sequential(*shortcut)

        self.apply(self._init_weights)

    def forward(self, x):
        raise NotImplementedError()
    
    def _init_weights(self, m):
        if isinstance(m, (CausalConv1d, Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    m.bias.data.fill_(value=0.0)
