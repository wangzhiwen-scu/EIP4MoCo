import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Optional, List
from torch import Tensor
import numpy as np
import warnings

class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


def computePatchNumAndSize(image_size, down_scale):
    N = image_size * image_size
    num_patches_base1 = int(np.prod(down_scale[0]) ** 1)
    num_patches_base2 = num_patches_base1 ** 2
    num_patches_base3 = num_patches_base1 ** 3
    num_patches_base4 = num_patches_base1 ** 4

    num_patches1 = N // num_patches_base1
    num_patches2 = N // num_patches_base2
    num_patches3 = N // num_patches_base3
    num_patches4 = N // num_patches_base4

    base = 2
    patch_size1 = image_size // base
    patch_size2 = patch_size1 // base
    patch_size3 = patch_size2 // base
    patch_size4 = patch_size3 // base

    return N, num_patches1, num_patches2, num_patches3, num_patches4, \
           patch_size1, patch_size2, patch_size3, patch_size4


# implement tf.gather.nd() in pytorch
def gather_nd(tensor, indexes, ndim):
    '''
    inputs = torch.randn(1, 3, 5)
    base = torch.arange(3)
    X_row = base.reshape(-1, 1).repeat(1, 5)
    lookup_sorted, indexes = torch.sort(inputs, dim=2, descending=True)
    print(inputs)
    print(indexes, indexes.shape)
    # print(gathered)
    print(gather_nd(inputs, indexes, [1, 2]))
    '''
    if len(ndim) == 2:
        base = torch.arange(indexes.shape[ndim[0]])
        row_index = base.reshape(-1, 1).repeat(1, indexes.shape[ndim[1]])
        gathered = tensor[..., row_index, indexes]
    elif len(ndim) == 1:
        base = torch.arange(indexes.shape[ndim[0]])
        gathered = tensor[..., base, indexes]
    else:
        raise NotImplementedError
    return gathered

class PrimalUpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 逐层回放
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = nn.Sequential(#nn.ModuleList([
            #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #     ResBlock(out_channels, kernel_size=3),#,
            #     ResBlock(out_channels, kernel_size=3)
            # )
            # ])
            self.conv = PrimalConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = PrimalConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, bs):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PrimalConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResBlock(nn.Module):
    def __init__(
            self, n_feats, kernel_size=3,
            bias=True, bn=False, act=nn.LeakyReLU(0.2, True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size,
                               padding=(kernel_size // 2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class PatchConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_stride=16, stride=16, in_chans=3, embed_dim=768, is_padding=False,
                 compatiable=False):
        super().__init__()

        if is_padding or patch_stride >= 3:
            if stride % 2 != 0 and img_size % 2 != 0:
                raise ValueError("size is not suitable")
            # 由输出和输入大小(stride)决定尺寸，而不是卷积核的大小，但padding不能太大，否则无意义
            o_size = img_size // stride
            padding = int(np.ceil(((o_size - 1) * stride + patch_stride - img_size) / 2))
            if padding >= o_size:
                raise ValueError("padding is too large")
        else:
            padding = 0
        img_size = _pair(img_size)
        patch_stride = _pair(patch_stride)
        stride = _pair(stride)

        self.img_size = img_size
        self.stride = stride
        assert img_size[0] % stride[0] == 0 and img_size[1] % stride[1] == 0, \
            f"img_size {img_size} should be divided by Conv2D stride {stride}."
        self.H, self.W = img_size[0] // patch_stride[0], img_size[1] // patch_stride[1]
        # self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_stride, stride=stride,
                              padding=padding)  # patch_stride
        self.norm = nn.LayerNorm(embed_dim)
        # self.pa = PatchifyAugment(False, self.H)
        self.comp = compatiable

    def forward(self, x, bs):
        B, C, H, W = x.shape
        p = B // bs

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.stride[0], W // self.stride[1]
        if len(x.shape) == 3:
            x = x.reshape(bs, p, H * W, -1)
            # x = x.reshape(bs*p, H, W, -1)

        return x, (H, W, x.shape[-1])


def compute_patches(img_size, kernel_size, stride, is_padding=False):
    nH = (img_size - kernel_size) // stride + 1
    # o_size = kernel_size, is_padding=False, 当为True时
    num_patch = nH ** 2
    if is_padding:
        # 由步长决定尺寸，而不是卷积核的大小，但padding不能太大，否则无意义
        o_size = img_size // stride
        if o_size <= 0:
            raise ValueError("when padding, stride > img_size, o_size: {}".format(o_size))

        padding = ((o_size - 1) * stride + kernel_size - img_size) // 2
        if padding < o_size and padding>0:
            nH = (img_size - o_size + 2 * padding) // stride + 1
            num_patch = nH ** 2
        else:
            raise ValueError("padding is too large or padding < 0, padding: {}".format(padding))
    else:
        padding = 0
    print("o_size = img_size // stride:", is_padding)
    print("img_size:", img_size, "kernel_size:", kernel_size)
    print("stride:", stride)
    print("num_patch: ", num_patch, "(nH,nW):", nH, nH)
    print("padding:", padding)

    # nH*nW, H, W, padding
    return num_patch, kernel_size, padding

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x