"""
This code is adapted from the original dac repository.
sources: 
    - https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/layers.py
    - https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm
import math


EPS = 1e-8 

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
        # nn.init.constant_(m.bias, 0)
        
        
class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y
    
    
class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)
    
    
class DenoisingBlock(nn.Module):
    def __init__(self, 
                 dim: int = 16
                #  stride: int = 1
                 ):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim, dilation=1),
            ResidualUnit(dim, dilation=3),
            ResidualUnit(dim, dilation=9),
            Snake1d(dim),
            WNConv1d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                # padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)

    
#################
"""
TCN Modules, not used.
"""


class DenoisingBlockTCN(nn.Module):
    # def __init__(self, dim, B, H, P, X, R):
    def __init__(self, dim_in, dim_out, strides_list):
        """
        Based on TCN architecture.
        N: Number of filters in autoencoder
        B: Number of channels in bottleneck 1 × 1-conv block
        H: Number of channels in convolutional blocks
        P: Kernel size in convolutional blocks
        X: Number of convolutional blocks in each repeat
        R: Number of repeats
        """
        """
        In our case...
        N = channel
        B = channel
        H = channel * 2
        P = kernel size = 7
        X = 3 : number of convolutional blocks in each repeat
        R = 2 : TCN block
        """
        super().__init__()
        N = dim_in
        B = dim_in
        # H = dim_in * 2
        H = B
        P = 7
        X = 3
        R = len(strides_list)
        
        # bottleneck = WNConv1d(N, B, kernel_size=1, bias=False)
        self.layer_norm = ChannelwiseLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1, bias=False)
        blocks_tcn = []
        blocks_downsample = []
        self.R = R
        for r in range(R):
            inner_blocks = []
            for x in range(X):
                dilation = 3**x
                padding = (P - 1) * dilation // 2
                inner_blocks.append(
                    TemporalBlock(B, H, P, stride=1, 
                                  padding=padding, dilation=dilation)
                )
            blocks_tcn.append(nn.Sequential(*inner_blocks))
            blocks_downsample.append(
                WNConv1d(B, B, kernel_size=2*strides_list[r],
                         stride=strides_list[r], padding=math.ceil(strides_list[r] / 2))
            )
            # blocks_tcn.append(
            #     WNConv1d(B, B, kernel_size=2*strides_list[r], 
            #              stride=strides_list[r], padding=math.ceil(strides_list[r] / 2))
            # )
            # repeats.append(nn.Sequential(*blocks))
        self.temporal_conv_net = nn.Sequential(*blocks_tcn)
        self.temporal_downsample = nn.Sequential(*blocks_downsample)
        # self.mask_conv = nn.Conv1d(B, dim_out, 1, bias=False)
        self.mask_conv = WNConv1d(B, dim_out, 1, bias=False)
        ## mean-only batch norm
        # self.net = nn.Sequential(
        #     layer_norm,
        #     bottleneck,
        #     temporal_conv_net,
        #     mask_conv
        # )
    
    def forward(self, x):
        """
        Args:
            x: [M, N, K]
        Returns:
            [M, N, K]
        """
        M, N, K = x.size()
        skip_connection = 0
        x = self.layer_norm(x)
        x = self.bottleneck(x)
        for ii in range(len(self.temporal_conv_net)):
            residual = self.temporal_conv_net[ii](x) ## slightly different with original ConvTasNet
            x = x + residual
            x = self.temporal_downsample[ii](x)
        x = self.mask_conv(x)
        # est_mask = torch.sigmoid(x)
        return x
        
        
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        # conv1x1 = WNConv1d(in_channels, out_channels, 1, bias=False)
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = GlobalLayerNorm(out_channels)
        # norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        # return out
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)
        
        
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        norm = GlobalLayerNorm(in_channels)
        prelu = nn.PReLU()
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.net = nn.Sequential(depthwise_conv, prelu, norm, pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)
    
    
class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y
    
# class ChannelWiseLayerNorm(nn.LayerNorm):
#     """
#     Channel wise layer normalization
#     """

#     def __init__(self, *args, **kwargs):
#         super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

#     def forward(self, x):
#         """
#         x: N x C x T
#         """
#         if x.dim() != 3:
#             raise RuntimeError("{} accept 3D tensor as input".format(
#                 self.__name__))
#         # N x C x T => N x T x C
#         x = torch.transpose(x, 1, 2)
#         # LN
#         x = super().forward(x)
#         # N x C x T => N x T x C
#         x = torch.transpose(x, 1, 2)
#         return x
    

class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y
    
    
    
class DenoisingBlockTCN2(nn.Module):
    # def __init__(self, dim, B, H, P, X, R):
    def __init__(self, dim_in, dim_out, R=4):
        """
        Based on TCN architecture.
        N: Number of filters in autoencoder
        B: Number of channels in bottleneck 1 × 1-conv block
        H: Number of channels in convolutional blocks
        P: Kernel size in convolutional blocks
        X: Number of convolutional blocks in each repeat
        R: Number of repeats
        """
        """
        In our case...
        N = channel
        B = channel
        H = channel * 2
        P = kernel size = 7
        X = 3 : number of convolutional blocks in each repeat
        R = 2 : TCN block
        """
        super().__init__()
        N = dim_in
        B = dim_in
        H = dim_in * 2
        # H = B
        P = 7
        X = 3
        # R = 4
        
        # bottleneck = WNConv1d(N, B, kernel_size=1, bias=False)
        self.layer_norm = ChannelwiseLayerNorm(N)
        self.bottleneck = nn.Conv1d(N, B, 1, bias=False)
        blocks_tcn = []
        self.R = R
        for r in range(R):
            inner_blocks = []
            for x in range(X):
                dilation = 3**x
                padding = (P - 1) * dilation // 2
                inner_blocks.append(
                    TemporalBlock(B, H, P, stride=1, 
                                  padding=padding, dilation=dilation)
                )
            blocks_tcn.append(nn.Sequential(*inner_blocks))
            # blocks_tcn.append(
            #     WNConv1d(B, B, kernel_size=2*strides_list[r], 
            #              stride=strides_list[r], padding=math.ceil(strides_list[r] / 2))
            # )
            # repeats.append(nn.Sequential(*blocks))
        self.temporal_conv_net = nn.Sequential(*blocks_tcn)
        # self.mask_conv = nn.Conv1d(B, dim_out, 1, bias=False)
        self.mask_conv = WNConv1d(B, dim_out, 1, bias=False)
        ## mean-only batch norm
        # self.net = nn.Sequential(
        #     layer_norm,
        #     bottleneck,
        #     temporal_conv_net,
        #     mask_conv
        # )
    
    def forward(self, x):
        """
        Args:
            x: [M, N, K]
        Returns:
            [M, N, K]
        """
        M, N, K = x.size()
        x = self.layer_norm(x)
        x = self.bottleneck(x)
        for ii in range(len(self.temporal_conv_net)):
            residual = self.temporal_conv_net[ii](x) ## slightly different with original ConvTasNet
            x = x + residual
        x = self.mask_conv(x) ## mask_conv(x): (M, )
        # est_mask = torch.sigmoid(x)
        return x
    
