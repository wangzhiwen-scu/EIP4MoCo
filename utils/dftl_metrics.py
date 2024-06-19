# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:
import torch
import numpy as np
import math
from torch import nn as nn
# from common.cal_ssim import SSIM

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def rgb2ycbcr(img, y_only=True):
    """metrics"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt


def quantize(img, rgb_range):
    pixel_range = 255.0 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range):
    """metrics"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    # .reshape((1, 1, 3)) / 256#
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)  # (1)
    if hr.size == 1:
        return 0
    if scale != 1:
        shave = scale
    else:
        shave = scale + 6
    if scale == 1:
        valid = diff
    else:
        valid = diff[..., shave:-shave, shave:-shave]
        # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.mean(pow(valid, 2), axis=[1, 2, 3]), axis=0)
    mse = np.mean(pow(valid, 2))
    if mse == 0:
        return 100
    try:
        psnr = -10 * math.log10(mse)
    except Exception:
        print(mse)

    return psnr


class PSNR_ycbcr(nn.Module):

    def __init__(self):
        super().__init__()
        self.gray_coeffs = torch.tensor([65.738, 129.057, 25.064],
                                        requires_grad=False).reshape((1, 3, 1, 1)) / 256

    def quantize(self, img, rgb_range):
        """metrics"""
        pixel_range = 255 / rgb_range
        img = torch.multiply(img, pixel_range)
        img = torch.clip(img, 0, 255)
        img = torch.round(img) / pixel_range
        return img

    @torch.no_grad()
    def forward(self, sr, hr, scale, rgb_range):
        """metrics"""
        sr = self.quantize(sr, rgb_range)
        gray_coeffs = self.gray_coeffs.to(sr.device)

        hr = hr.float()
        sr = sr.float()
        diff = (sr - hr) / rgb_range

        diff = torch.multiply(diff, gray_coeffs).sum(1)
        if hr.size == 1:
            return 0
        if scale != 1:
            shave = scale
        else:
            shave = scale + 6
        if scale == 1:
            valid = diff
        else:
            valid = diff[..., shave:-shave, shave:-shave]
        mse = torch.mean(torch.pow(valid, 2))
        return -10 * torch.log10(mse)


def sub_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x / 255.0


def add_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3):
        super(SSIM, self).__init__()
        self.window_size = win_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(win_size, self.channel, win_sigma)
        self.win_sigma = win_sigma

    def forward(self, img1, img2):
        # print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.win_sigma)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, win_size=11, data_range=1, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, win_size, channel, size_average)