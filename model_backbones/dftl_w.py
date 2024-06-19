import math
import os
from functools import partial

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('.') 
from layers.dftl_base import SetCriterion
from torch import optim

from layers.dftl_base import PatchMergeModule
from model_backbones.dftl import DerainModel
from layers.dftl_modules.module import *
from layers.dftl_modules.self_attn_module import *
from layers.dftl_modules.loss import BCMSLoss

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


class CheapLP(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(CheapLP, self).__init__()
        self.oup = oup
        init_channels = oup // ratio
        new_channels = init_channels * ratio

        self.vit_mlp = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
        )

        self.dw_mlp = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
        )

    def forward(self, x):
        x1 = self.vit_mlp(x)
        x2 = self.dw_mlp(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

class FeedForward(nn.Module):
    def __init__(self, patch_size, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        # self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.project_in = CheapLP(dim, hidden_features * 2)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        # self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.project_out = CheapLP(hidden_features, dim)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return x

class SwinBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, norm_dim, num_heads, window_size=6, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, compatiable=True, hybrid_ffn=True):
        super().__init__()
        self.__class__.__name__ = 'SwinTEB'
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim if compatiable else norm_dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = FeedForward(None, dim, mlp_ratio, False)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, H=None, W=None):# H, W
        B, p, L, C = x.shape
        # B, H, W, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        H = W = self.input_resolution
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == H:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:

            attn_windows = self.attn(x_windows, mask=self.calculate_mask((H, W)).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, 1, H*W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2) + self.ffn(self.norm2(x).view(B, H, W, C).permute(0, 3, 1, 2))

        return x.permute(0, 2, 3, 1).view(B, 1, H*W, C)



class DFTLW(PatchMergeModule):
    def __init__(self, args, image_size, down_scale, patch_stride, stride, windows_size, in_channels, out_channels,
                 hidden_dim, num_heads, mlp_ratios, depths, sr_ratio,
                 norm_pre=False, bilinear=True):
        super(DFTLW, self).__init__(bs_axis_merge=False)

        self.args = args
        N, num_patches1, num_patches2, num_patches3, \
        num_patches4, patch_size1, patch_size2, \
        patch_size3, patch_size4 = computePatchNumAndSize(image_size, down_scale)

        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim[0], kernel_size=3, padding=1),
                ResBlock(hidden_dim[0], kernel_size=5),
                ResBlock(hidden_dim[0], kernel_size=5)
            ) for _ in range(1)
        ])

        self.patch_embed1 = PatchConvEmbed(img_size=image_size, patch_stride=patch_stride[0], in_chans=hidden_dim[0], stride=stride[0],
                                           embed_dim=hidden_dim[1], compatiable=True)
        self.patch_embed2 = PatchConvEmbed(img_size=image_size // 2, patch_stride=patch_stride[1],stride=stride[1],
                                           in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2], compatiable=True)
        self.patch_embed3 = PatchConvEmbed(img_size=image_size // 4, patch_stride=patch_stride[2],stride=stride[2],
                                           in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3], compatiable=True)
        self.patch_embed4 = PatchConvEmbed(img_size=image_size // 8, patch_stride=patch_stride[3],stride=stride[3],
                                           in_chans=hidden_dim[3],
                                           embed_dim=hidden_dim[4], compatiable=True)

        self.down1 = nn.ModuleList([SwinBlock(input_resolution=patch_size1, window_size=windows_size[0], norm_dim=[1, num_patches1, hidden_dim[1]],
                                           dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                           qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                                    for i in range(depths[0])])

        self.down2 = nn.ModuleList([SwinBlock(input_resolution=patch_size2, window_size=windows_size[1], norm_dim=[1, num_patches2, hidden_dim[2]],
                                           dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                                           qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                                    for i in range(depths[1])])

        self.down3 = nn.ModuleList([SwinBlock(input_resolution=patch_size3, window_size=windows_size[2], norm_dim=[1, num_patches3, hidden_dim[3]],
                                           dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                           qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                                    for i in range(depths[2])])

        self.down4 = nn.ModuleList([SwinBlock(input_resolution=patch_size4, window_size=windows_size[3], norm_dim=[1, num_patches4, hidden_dim[4]],
                                           dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                           qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6))
                                    for i in range(depths[3])])




        # self.norm = nn.LayerNorm([hidden_dim[1], image_size, image_size], eps=1e-6)
        self.norm = nn.LayerNorm(hidden_dim[1], eps=1e-6)

        factor = 2 if bilinear else 1
        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], hidden_dim[3], bilinear)
        self.up2 = PrimalUpBlock(hidden_dim[3] + hidden_dim[2], hidden_dim[2], bilinear)
        self.up3 = PrimalUpBlock(hidden_dim[2] + hidden_dim[1], hidden_dim[1], bilinear)
        self.up4 = PrimalUpBlock(hidden_dim[1] + hidden_dim[0], hidden_dim[0], bilinear)
        self.tail = OutConv(hidden_dim[0], out_channels)

        # self.apply(self._init_weights)

    def _init_weights(self, m):

        if isinstance(m, nn.Conv2d):
            # print("1:", m.groups)
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # nn.init.constant_(m.bias, 0.)
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_post(self, I):

        bs, C, H, W = I.shape
        # x = self.pa(x)

        # N,C,H,W
        x0 = I
        for blk in self.head:
            x0 = blk(x0)

        x, (H, W, D) = self.patch_embed1(x0, bs)  # x
        # x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, H, W)
        # [B, P, N, C]->[BP, H, W, C]，用于patch_embed
        x1 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        # print(x1.shape)
        # stage 2 -128
        x, (H, W, D) = self.patch_embed2(x1, bs)
        # x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, H, W)
        x2 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        # print(x2.shape)
        # stage 3 -320
        x, (H, W, D) = self.patch_embed3(x2, bs)
        # x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, H, W)
        x3 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        # print(x3.shape)
        # stage 4 -512
        x, (H, W, D) = self.patch_embed4(x3, bs)

        # x = x + self.pos_embed4
        for blk in self.down4:
            x = blk(x, H, W)
        x4 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()
        x = self.up1(x4, x3, bs)
        x = self.up2(x, x2, bs)
        x = self.up3(x, x1, bs)
        x = self.up4(x, x0, bs)

        logits = self.tail(x) + I

        return logits

    def forward(self, x):
        return self.forward_post(x)

    def train_step(self, *args, **kwargs):  # data, *args, **kwargs):

        return self(*args, **kwargs)

    def val_step(self, *args, **kwargs):

        return self.forward_chop(*args, **kwargs)

    # def train_step(self, batch, *args, **kwargs):
    #     O, B = batch['O'].cuda(), batch['B'].cuda()
    #     derain = self(sub_mean(O))
    #
    #     loss = self.criterion(derain, sub_mean(B), *args, **kwargs)
    #     derain = add_mean(derain)
    #     with torch.no_grad():
    #         loss.update(psnr=reduce_mean(self.psnr(derain, B * 255, 4, 255.0)))
    #         loss.update(ssim=reduce_mean(self.ssim(derain / 255.0, B)))
    #
    #     return derain, loss
    #
    # def val_step(self, batch, saved_path):
    #
    #     metrics = {}
    #     O, B = batch['O'].cuda(), batch['B'].cuda()
    #     samples = sub_mean(O)
    #     derain = self.forward_chop(samples)
    #     pred = quantize(add_mean(derain), 255)
    #     normalized = pred[0]
    #     tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    #
    #     with torch.no_grad():
    #         metrics.update(ssim=reduce_mean(torch.mean(self.ssim(pred / 255.0, B))))
    #         metrics.update(psnr=reduce_mean(self.psnr(pred, B * 255.0, 4, 255.0)))
    #
    #     imageio.imwrite(os.path.join(saved_path, batch['filename'][0]+'.png'),#, '_', str(int(metrics['psnr'].item())), '.png'])),
    #                     tensor_cpu.numpy())
    #
    #     return metrics

class build_DFTLW(DerainModel, name='DFTLW'):

    def __call__(self, args):

        scheduler = None
        scale = 2
        weight_dict = {'loss': 1}
        # losses = {'loss': nn.L1Loss().cuda()}
        losses = {'loss': BCMSLoss().cuda()}
        criterion = SetCriterion(losses, weight_dict)
        model = DFTLW(args, image_size=args.patch_size[0], in_channels=3, out_channels=3,
                             down_scale=[[scale, scale], [scale, scale], [scale, scale], [scale, scale]],
                             patch_stride=[scale, scale, scale, scale], windows_size=[4, 4, 4, 4], stride=[2] * 4,
                             hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                             depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
        model.set_metrics(criterion)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  ## optimizer 1: Adam

        return model, criterion, optimizer, scheduler
    
if __name__ == "__main__":

    scale =2
    patch_size = 32
    model = DFTLW(None, image_size=patch_size, in_channels=1, out_channels=1,
                            down_scale=[[scale, scale], [scale, scale], [scale, scale], [scale, scale]],
                            patch_stride=[scale, scale, scale, scale], windows_size=[4, 4, 4, 4], stride=[2] * 4,
                            hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                            depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    
    inpu = torch.randn(1,1,256,256).cuda()
    out = model(inpu)
    print(out.shape)