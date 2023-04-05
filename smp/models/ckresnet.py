import torch
import ckconv.nn
from functools import partial
from .residual_block import ResidualBlockBase
from timm.models.layers import trunc_normal_

# typing
from omegaconf import OmegaConf
from typing import Tuple, Union
from ckconv.nn import CKConv, CausalConv1d
from torch.nn import Conv1d, Conv2d


class ResNetBlock(ResidualBlockBase):
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
        Creates a Residual Block as in the original ResNet paper (He et al., 2016)

        input
         | ---------------|
         CKConv           |
         LayerNorm        |
         ReLU             |
         DropOut          |
         |                |
         CKConv           |
         LayerNorm        |
         |                |
         + <--------------|
         |
         ReLU
         |
         output
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ConvType=ConvType,
            NormType=NormType,
            NonlinearType=NonlinearType,
            LinearType=LinearType,
            dropout=dropout,
            dim_linear=dim_linear,
            small_kernel_size=small_kernel_size
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        # Following Sosnovik et al. 2020, dropout placed after first ReLU.
        out = self.dp(self.nonlinear(self.norm1(self.cconv1(x)) + self.norm1s(self.cconv1s(x))))
        out = self.nonlinear(self.norm2(self.cconv2(out)) + self.norm2s(self.cconv2s(out)) + shortcut)
        return out


class Img_ResNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__()

        # Unpack arguments from net_config
        hidden_channels = net_config.no_hidden
        no_blocks = net_config.no_blocks
        norm = net_config.norm
        dropout = net_config.dropout
        block_width_factors = net_config.block_width_factors
        nonlinearity = net_config.nonlinearity

        # Unpack conv_config
        conv_type = conv_config.type
        conv_use_fft = conv_config.use_fft
        conv_horizon = conv_config.horizon
        conv_padding = conv_config.padding
        conv_stride = conv_config.stride
        conv_bias = conv_config.bias
        conv_small_kernel_size = conv_config.small_kernel_size

        # Unpack kernel_config
        kernel_dim_linear = kernel_config.dim_linear
        kernel_n_points = kernel_config.n_points
        kernel_radius = kernel_config.radius
        kernel_coord_std = kernel_config.coord_std

        # Define partials for types of convs
        if conv_type == "CKConv":
            ConvType = partial(
                ckconv.nn.CKConv,
                horizon=conv_horizon,
                kernel_dim_linear=kernel_dim_linear,
                kernel_n_points=kernel_n_points,
                kernel_radius=kernel_radius,
                kernel_coord_std=kernel_coord_std,
                conv_use_fft=conv_use_fft,
                conv_bias=conv_bias,
                conv_padding=conv_padding,
                conv_stride=1,
            )
        elif conv_type == "Conv":
            ConvType = partial(
                getattr(torch.nn, f"Conv{kernel_dim_linear}d"),
                kernel_size=int(conv_horizon),
                padding=conv_padding,
                stride=conv_stride,
                bias=conv_bias,
            )
        else:
            raise NotImplementedError(f"conv_type = {conv_type}")
        # -------------------------

        # Define NormType
        NormType = {
            "BatchNorm": getattr(torch.nn, f"BatchNorm{kernel_dim_linear}d"),
            "LayerNorm": ckconv.nn.LayerNorm,
        }[norm]

        NonlinearType = {"ReLU": torch.nn.ReLU, "LeakyReLU": torch.nn.LeakyReLU}[
            nonlinearity
        ]

        # Define LinearType
        LinearType = getattr(ckconv.nn, f"Linear{kernel_dim_linear}d")

        # Create Input Layers
        self.cconv1 = ConvType(in_channels=in_channels, out_channels=hidden_channels)
        if kernel_dim_linear == 1:
            self.cconv1s = CausalConv1d(in_channels=in_channels, out_channels=hidden_channels,
                                                kernel_size=conv_small_kernel_size)
        elif kernel_dim_linear == 2:
            self.cconv1s = Conv2d(in_channels=in_channels, out_channels=hidden_channels,
                                    kernel_size=conv_small_kernel_size, padding=conv_small_kernel_size//2)
        trunc_normal_(self.cconv1s.weight, std=.02)
        if hasattr(self.cconv1s, 'bias'):
            print("head bias init")
            self.cconv1s.bias.data.fill_(value=0.0)
        self.norm1 = NormType(hidden_channels)
        self.norm1s = NormType(hidden_channels)
        self.nonlinear = NonlinearType()

        # Create Blocks
        # -------------------------
        # Create vector of width_factors:
        # If value is zero, then all values are one
        if block_width_factors[0] == 0.0:
            width_factors = (1,) * no_blocks
        else:
            width_factors = [
                (factor,) * n_blcks
                for factor, n_blcks in ckconv.utils.pairwise_iterable(
                    block_width_factors
                )
            ]
            width_factors = [
                factor for factor_tuple in width_factors for factor in factor_tuple
            ]

        if len(width_factors) != no_blocks:
            raise ValueError(
                "The size of the width_factors does not matched the number of blocks in the network."
            )

        blocks = []
        for i in range(no_blocks):
            print(f"Block {i}/{no_blocks}")

            if i == 0:
                input_ch = hidden_channels
                hidden_ch = int(hidden_channels * width_factors[i])
            else:
                input_ch = int(hidden_channels * width_factors[i - 1])
                hidden_ch = int(hidden_channels * width_factors[i])

            blocks.append(
                ResNetBlock(
                    input_ch,
                    hidden_ch,
                    ConvType=ConvType,
                    NonlinearType=NonlinearType,
                    NormType=NormType,
                    LinearType=LinearType,
                    dropout=dropout,
                    dim_linear=kernel_dim_linear,
                    small_kernel_size=conv_small_kernel_size
                )
            )

        self.blocks = torch.nn.Sequential(*blocks)
        # -------------------------

        # Define Output Layers:
        # -------------------------
        # calculate output channels of blocks
        if block_width_factors[0] == 0.0:
            final_no_hidden = hidden_channels
        else:
            final_no_hidden = int(hidden_channels * block_width_factors[-2])
        # instantiate last layer
        self.out_layer = LinearType(
            in_channels=final_no_hidden, out_channels=out_channels
        )
        # Initialize finallyr
        torch.nn.init.kaiming_normal_(self.out_layer.weight)
        self.out_layer.bias.data.fill_(value=0.0)
        # -------------------------

        # Save variables in self
        self.dim_linear = kernel_dim_linear

    def forward(self, x):
        # First layers
        out = self.nonlinear(self.norm1(self.cconv1(x)) + self.norm1s(self.cconv1s(x)))
        # Blocks
        out = self.blocks(out)
        # Pool
        out = torch.nn.functional.adaptive_avg_pool2d(
            out,
            (1,) * self.dim_linear,
        )
        # Final layer
        out = self.out_layer(out)
        return out.squeeze()


class SeqData_ResNet(Img_ResNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        net_config: OmegaConf,
        kernel_config: OmegaConf,
        conv_config: OmegaConf,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            net_config=net_config,
            kernel_config=kernel_config,
            conv_config=conv_config,
        )

    def forward(self, x):
        # First layers
        out = torch.relu(self.norm1(self.cconv1(x)) + self.norm1s(self.cconv1s(x)))
        # Blocks
        out = self.blocks(out)
        # Final layer on last sequence element
        out = self.out_layer(out[:, :, -1:])
        return out.squeeze(-1)