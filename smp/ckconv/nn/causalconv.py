import torch
import torch.nn
import ckconv.nn.functional as ckconv_f


class CausalConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
    
    def forward(self, x):
        if hasattr(self, 'bias'):
            return ckconv_f.causal_conv(x, self.weight, bias=self.bias)
        else:
            return ckconv_f.causal_conv(x, self.weight, bias=None)
