import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

from depthwise_conv2d_implicit_gemm import _DepthWiseConv2dImplicitGEMMFP16, _DepthWiseConv2dImplicitGEMMFP32


def rel_pos(kernel_size):
    tensors = [torch.linspace(-1, 1, steps=kernel_size) for _ in range(2)]
    kernel_coord = torch.stack(torch.meshgrid(*tensors), dim=-0)
    kernel_coord = kernel_coord.unsqueeze(0)
    return kernel_coord


class SMPConv(nn.Module):
    def __init__(self, planes, kernel_size, n_points, stride, padding, groups):
        super().__init__()

        self.planes = planes
        self.kernel_size = kernel_size
        self.n_points = n_points
        self.init_radius = 2 * (2/kernel_size)

        # kernel_coord
        kernel_coord = rel_pos(kernel_size)
        self.register_buffer('kernel_coord', kernel_coord)

        # weight_coord
        weight_coord = torch.empty(1, n_points, 2)
        nn.init.trunc_normal_(weight_coord, std=0.2, a=-1., b=1.)
        self.weight_coord = nn.Parameter(weight_coord)

        self.radius = nn.Parameter(torch.empty(1, n_points).unsqueeze(-1).unsqueeze(-1))
        self.radius.data.fill_(value=self.init_radius)

        # weight
        weights = torch.empty(1, planes, n_points)
        trunc_normal_(weights, std=.02)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        kernels = self.make_kernels().unsqueeze(1)
        x = x.contiguous()
        kernels = kernels.contiguous()

        if x.dtype == torch.float32:
            x = _DepthWiseConv2dImplicitGEMMFP32.apply(x, kernels)
        elif x.dtype == torch.float16:
            x = _DepthWiseConv2dImplicitGEMMFP16.apply(x, kernels)
        else:
            raise TypeError("Only support fp32 and fp16, get {}".format(x.dtype))
        return x        

    def make_kernels(self):
        diff = self.weight_coord.unsqueeze(-2) - self.kernel_coord.reshape(1,2,-1).transpose(1,2)  # [1, n_points, kernel_size^2, 2]
        diff = diff.transpose(2,3).reshape(1, self.n_points, 2, self.kernel_size, self.kernel_size)
        diff = F.relu(1 - torch.sum(torch.abs(diff), dim=2) / self.radius)  # [1, n_points, kernel_size, kernel_size]
        
        # Apply weighted diff for average weighted kernel
        non_zero = (diff != 0) # [1, n_points, kernel_size, kernel_size]
        count_weight = 1 / (torch.sum(non_zero, dim=1, keepdim=True) + 1e-6)  # [1, 1, kernel_size, kernel_size]
        weighted_diff = count_weight * diff  # [1, n_points, kernel_size, kernel_size]

        kernels = torch.matmul(self.weights, diff.reshape(1, self.n_points, -1)) # [1, planes, kernel_size*kernel_size]
        kernels = kernels.reshape(1, self.planes, *self.kernel_coord.shape[2:]) # [1, planes, kernel_size, kernel_size]
        kernels = kernels.squeeze(0)
        kernels = torch.flip(kernels.permute(0,2,1), dims=(1,))
        return kernels
    
    def radius_clip(self, min_radius=1e-3, max_radius=1.):
        r = self.radius.data
        r = r.clamp(min_radius, max_radius)
        self.radius.data = r


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, n_points=None):
    if n_points != None and in_channels == out_channels and out_channels == groups and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        print("SMPConv")
        return SMPConv(in_channels, kernel_size, n_points, stride, padding, groups)
    else:
        print("Original convolution")
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)


use_sync_bn = False

def enable_sync_bn():
    global use_sync_bn
    use_sync_bn = True


def get_bn(channels):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False, 
                                         n_points=n_points))
    result.add_module('bn', get_bn(out_channels))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, n_points=None):
    if padding is None:
        padding = kernel_size // 2
    result = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, groups=groups, dilation=dilation,
                                         n_points=n_points)
    result.add_module('nonlinear', nn.ReLU())
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class SMPCNN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups, n_points=None, n_points_divide=4):
        super().__init__()
        self.kernel_size = kernel_size
        if n_points == None:
            n_points = int((kernel_size**2) // n_points_divide)

        padding = kernel_size // 2
        self.smp = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, dilation=1, groups=groups, n_points=n_points)
        
        self.small_kernel = 5
        self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=self.small_kernel,
                                   stride=stride, padding=self.small_kernel//2, groups=groups)

    def forward(self, inputs):
        out = self.smp(inputs)
        out += self.small_conv(inputs)
        return out


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = get_bn(in_channels)
        self.pw1 = conv_bn(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.pw2 = conv_bn(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class SMPBlock(nn.Module):

    def __init__(self, in_channels, dw_channels, lk_size, drop_path, n_points=None, n_points_divide=4):
        super().__init__()
        self.pw1 = conv_bn_relu(in_channels, dw_channels, 1, 1, 0, groups=1)
        self.pw2 = conv_bn(dw_channels, in_channels, 1, 1, 0, groups=1)
        self.large_kernel = SMPCNN(in_channels=dw_channels, out_channels=dw_channels, kernel_size=lk_size,
                                  stride=1, groups=dw_channels, n_points=n_points, n_points_divide=n_points_divide)
        self.lk_nonlinear = nn.ReLU()
        self.prelkb_bn = get_bn(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class SMPStage(nn.Module):

    def __init__(self, channels, num_blocks, stage_lk_size, drop_path,
                 n_points=None, n_points_divide=4,
                 dw_ratio=1, ffn_ratio=2,
                 use_checkpoint=False,      # train with torch.utils.checkpoint to save memory
                 norm_intermediate_features=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        blks = []
        for i in range(num_blocks):
            block_drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path
            replk_block = SMPBlock(in_channels=channels, dw_channels=int(channels * dw_ratio), lk_size=stage_lk_size,
                                    drop_path=block_drop_path, n_points=n_points, n_points_divide=n_points_divide)
            convffn_block = ConvFFN(in_channels=channels, internal_channels=int(channels * ffn_ratio), out_channels=channels,
                                    drop_path=block_drop_path)
            blks.append(replk_block)
            blks.append(convffn_block)
        self.blocks = nn.ModuleList(blks)
        if norm_intermediate_features:
            self.norm = get_bn(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)   # Save training memory
            else:
                x = blk(x)
        return x


class SMPConvNet(nn.Module):

    def __init__(self, large_kernel_sizes, layers, channels, drop_path_rate,
                 dw_ratio=1, ffn_ratio=4, in_channels=3, num_classes=1000, out_indices=None,
                 n_points_divide=4,
                 use_checkpoint=False,
                 use_sync_bn=True,
                 norm_intermediate_features=False
                 ):
        super().__init__()

        if num_classes is None and out_indices is None:
            raise ValueError('must specify one of num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and out_indices is not None:
            raise ValueError('cannot specify both num_classes (for pretraining) and out_indices (for downstream tasks)')
        elif num_classes is not None and norm_intermediate_features:
            raise ValueError('for pretraining, no need to normalize the intermediate feature maps')
        self.out_indices = out_indices
        if use_sync_bn:
            enable_sync_bn()

        base_width = channels[0]
        self.use_checkpoint = use_checkpoint
        self.norm_intermediate_features = norm_intermediate_features
        self.num_stages = len(layers)
        self.stem = nn.ModuleList([
            conv_bn_relu(in_channels=in_channels, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=1, padding=1, groups=base_width),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=1, stride=1, padding=0, groups=1),
            conv_bn_relu(in_channels=base_width, out_channels=base_width, kernel_size=3, stride=2, padding=1, groups=base_width)])
        # stochastic depth. We set block-wise drop-path rate. The higher level blocks are more likely to be dropped. This implementation follows Swin.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            layer = SMPStage(channels=channels[stage_idx], num_blocks=layers[stage_idx],
                            stage_lk_size=large_kernel_sizes[stage_idx],
                            n_points_divide=n_points_divide,
                            drop_path=dpr[sum(layers[:stage_idx]):sum(layers[:stage_idx + 1])],
                            dw_ratio=dw_ratio, ffn_ratio=ffn_ratio,
                            use_checkpoint=use_checkpoint,
                            norm_intermediate_features=norm_intermediate_features)
            self.stages.append(layer)
            if stage_idx < len(layers) - 1:
                transition = nn.Sequential(
                    conv_bn_relu(channels[stage_idx], channels[stage_idx + 1], 1, 1, 0, groups=1),
                    conv_bn_relu(channels[stage_idx + 1], channels[stage_idx + 1], 3, stride=2, padding=1, groups=channels[stage_idx + 1]))
                self.transitions.append(transition)

        if num_classes is not None:
            self.norm = get_bn(channels[-1])
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(channels[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward_features(self, x):
        x = self.stem[0](x)
        for stem_layer in self.stem[1:]:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(stem_layer, x)     # save memory
            else:
                x = stem_layer(x)

        if self.out_indices is None:
            #   Just need the final output
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return x
        else:
            #   Need the intermediate feature maps
            outs = []
            for stage_idx in range(self.num_stages):
                x = self.stages[stage_idx](x)
                if stage_idx in self.out_indices:
                    outs.append(self.stages[stage_idx].norm(x))
                if stage_idx < self.num_stages - 1:
                    x = self.transitions[stage_idx](x)
            return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.out_indices:
            return x
        else:
            x = self.norm(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.head(x)
            return x
    
    def clipping(self):
        for m in self.modules():
            if hasattr(m, 'radius_clip'):
                m.radius_clip()

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()

    #   If your framework cannot automatically fuse BN for inference, you may do it manually.
    #   The BNs after and before conv layers can be removed.
    #   No need to call this if your framework support automatic BN fusion.
    def deep_fuse_BN(self):
        for m in self.modules():
            if not isinstance(m, nn.Sequential):
                continue
            if not len(m) in [2, 3]:  # Only handle conv-BN or conv-BN-relu
                continue
            #   If you use a custom Conv2d impl, assume it also has 'kernel_size' and 'weight'
            if hasattr(m[0], 'kernel_size') and hasattr(m[0], 'weight') and isinstance(m[1], nn.BatchNorm2d):
                conv = m[0]
                bn = m[1]
                fused_kernel, fused_bias = fuse_bn(conv, bn)
                fused_conv = get_conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size,
                                        stride=conv.stride,
                                        padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True)
                fused_conv.weight.data = fused_kernel
                fused_conv.bias.data = fused_bias
                m[0] = fused_conv
                m[1] = nn.Identity()


@register_model
def SMPConv_T(pretrained=False, **kwargs):
    model = SMPConvNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,8,2], channels=[96,192,384,768],
                       n_points_divide=4, drop_path_rate=0.1)
    return model


@register_model
def SMPConv_B(pretrained=False, **kwargs):
    model = SMPConvNet(large_kernel_sizes=[31,29,27,13], layers=[2,2,20,2], channels=[128,256,512,1024],
                       n_points_divide=4, drop_path_rate=0.5)
    return model


if __name__ == "__main__":
    model = SMPConv_B().cuda()
    params = 0
    for n, p in model.named_parameters():
        params += p.numel()
    print(params)

    x = torch.randn(2, 3, 224, 224).cuda()
    y = model(x)
    print("process end")