import copy
import math

import torch
import torch.nn
from timm.models.layers import trunc_normal_

import ckconv
import ckconv.nn.functional as ckconv_F
from ckconv.utils.grids import rel_positions_grid


class CKConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        horizon: int,
        kernel_dim_linear = 2,
        kernel_n_points=36,
        kernel_radius=0.002,
        kernel_coord_std=0.1,
        conv_use_fft = False,
        conv_bias = True,
        conv_padding = "same",
        conv_stride = 1,
    ):
        """
        Continuous Kernel Convolution.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels produced by the convolution
        :param horizon: Maximum kernel size. Recommended to be odd and cover the entire image.
        :param kernel_dim_linear: Dimensionality of the input signal, e.g. 2 for images.
        :param conv_use_fft: Whether to use FFT implementation of convolution.
        :param conv_bias: Whether to use bias in kernel generator.
        :param conv_padding: Padding strategy for convolution.
        :param conv_stride: Stride applied in convolution.
        """

        super().__init__()



        # Create the kernel
        self.Kernel = ckconv.nn.ck.SMPKernel(
            dim_linear=kernel_dim_linear,
            in_channels=in_channels,
            out_channels=out_channels,
            n_points=kernel_n_points,
            radius=kernel_radius,
            coord_std=kernel_coord_std,
        )

        if conv_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            self.bias.data.fill_(value=0.0)
        else:
            self.bias = None

        # Save arguments in self
        # ---------------------
        # Non-persistent values
        self.padding = conv_padding
        self.stride = conv_stride
        self.rel_positions = None
        self.kernel_dim_linear = kernel_dim_linear
        self.horizon = horizon
        self.use_fftconv = conv_use_fft

        # Variable placeholders
        self.register_buffer("train_length", torch.zeros(1).int(), persistent=True)
        self.register_buffer("conv_kernel", torch.zeros(in_channels), persistent=False)

        # Define convolution type
        conv_type = "conv"
        if conv_use_fft:
            conv_type = "fft" + conv_type
        if kernel_dim_linear == 1:
            conv_type = "causal_" + conv_type
        self.conv = getattr(ckconv_F, conv_type)


    def forward(self, x):
        # Construct kernel
        x_shape = x.shape

        rel_pos = self.handle_rel_positions(x)
        conv_kernel = self.Kernel(rel_pos).view(-1, x_shape[1], *rel_pos.shape[2:])

        # For computation of "weight_decay"
        self.conv_kernel = conv_kernel

        return self.conv(x, conv_kernel, self.bias)

    def handle_rel_positions(self, x):
        """
        Handles the vector or relative positions which is given to KernelNet.
        """
        if self.rel_positions is None:
            if self.train_length[0] == 0:

                # Decide the extend of the rel_positions vector
                if self.horizon == "full":
                    self.train_length[0] = (2 * x.shape[-1]) - 1
                elif self.horizon == "same":
                    self.train_length[0] = x.shape[-1]
                elif int(self.horizon) % 2 == 1:
                    # Odd number
                    self.train_length[0] = int(self.horizon)
                else:
                    raise ValueError(
                        f"The horizon argument of the operation must be either 'full', 'same' or an odd number in string format. Current value: {self.horizon}"
                    )

            # Creates the vector of relative positions.
            rel_positions = rel_positions_grid(
                grid_sizes=self.train_length.repeat(self.kernel_dim_linear)
            ).unsqueeze(0)
            self.rel_positions = rel_positions.to(x.device)
            # -> With form: [batch_size=1, dim, x_dimension, y_dimension, ...]

        return self.rel_positions
