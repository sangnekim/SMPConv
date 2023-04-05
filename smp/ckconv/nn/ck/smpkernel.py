import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_


class SMPKernel(torch.nn.Module):
    def __init__(
        self,
        dim_linear: int, # seq: 1, img: 2
        in_channels: int,
        out_channels: int,
        n_points: int,
        radius: float,
        coord_std: float,
    ):
        super().__init__()

        self.dim_linear = dim_linear
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_points = n_points
        self.init_radius = radius

        # weight coord
        weight_coord = torch.empty(out_channels, self.n_points, dim_linear)
        if dim_linear == 1:  # causal 1d conv
            nn.init.trunc_normal_(weight_coord, std=coord_std, a=-1., b=0)
            weight_coord *= 2
            weight_coord += torch.tensor([1.]).unsqueeze(0).unsqueeze(0)
        elif dim_linear == 2:
            nn.init.trunc_normal_(weight_coord, std=coord_std, a=-1., b=1)
        self.weight_coord = torch.nn.Parameter(weight_coord)
        
        # radius
        r = torch.empty(out_channels, n_points)        
        for _ in range(dim_linear):
            r = r.unsqueeze(-1)
        self.radius = torch.nn.Parameter(r)
        self.radius.data.fill_(value=self.init_radius)

        # weight
        weights = torch.randn(out_channels, self.in_channels, self.n_points)
        trunc_normal_(weights, std=.02)
        self.weights = torch.nn.Parameter(weights)
 
    def forward(self, x):
        diff = self.weight_coord.unsqueeze(-2) - x.reshape(1, self.dim_linear, -1).transpose(1,2)
        diff = diff.transpose(2,3).reshape(self.out_channels, self.n_points, self.dim_linear, *x.shape[2:])
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)
        # Apply weighted diff for average weighted kernel
        non_zero = (diff != 0)
        count_weight = 1 / (torch.sum(non_zero, dim=1, keepdim=True) + 1e-6)
        weighted_diff = count_weight * diff
        kernels = torch.matmul(self.weights, weighted_diff.reshape(self.out_channels, self.n_points, -1))
        kernels = kernels.reshape(self.out_channels, self.in_channels, *x.shape[2:])
        if self.dim_linear == 2:
            kernels = torch.flip(kernels.transpose(-2,-1), dims=(2,))
        return kernels
    
    def radius_clip(self, min_radius=0.0001, max_radius=1.):
        r = self.radius.data
        r = r.clamp(min_radius, max_radius)
        self.radius.data = r