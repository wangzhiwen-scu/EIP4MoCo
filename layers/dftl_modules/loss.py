# GPL License
# Copyright (C) 2022 , UESTC
# All Rights Reserved 
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference: 
import torch
from utils.dftl_metrics import SSIM


class BCMSLoss(torch.nn.Module):

    def __init__(self, reduction='none'):
        super(BCMSLoss, self).__init__()

        self.reduction = reduction
        self.l1_loss = torch.nn.L1Loss(reduction=reduction)
        self.l2_loss = torch.nn.MSELoss(reduction=reduction)
        self._ssim = SSIM(size_average=False, data_range=1)
        self.ind = 1

    def forward(self, x, gt):

        ind = self.ind
        # bce = self.bce_mse(x, gt)
        l1_loss = self.l1_loss(x, gt)
        l2_loss = self.l2_loss(x, gt)
        _ssim = self._ssim(x, gt)
        _ssim_loss = 1 - _ssim

        with torch.no_grad():
            w_1 = torch.mean(l1_loss)

            # pred = torch.clamp(x, 0, 1)
            # l2 = self.l2_loss(x, gt)
            w_2 = torch.mean(l2_loss)

            w_s = torch.mean(_ssim_loss)
            ssim_m = torch.mean(_ssim)
            # w_bce = torch.mean(bce)

        # loss = _ssim_loss.reshape(-1, 1, 1, 1) * \
        #        ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** (1 / ind)  # 119
        # loss = l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + 1e-8) ** ind + \
        # loss = _ssim_loss.reshape(-1, 1, 1, 1) * ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind  # 让l2朝更多方向解析，却增强l1 82

        # loss = _ssim_loss.reshape(-1, 1, 1, 1) * (
        #         (l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind + \
        #        l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + (
        #         bce / w_bce) ** ind + 1e-8) ** ind

        loss = _ssim_loss.reshape(-1, 1, 1, 1) * (
                (l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + 1e-8) ** ind + \
               l1_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l2_loss / w_2) ** ind + 1e-8) ** ind

        loss = torch.mean(loss)
        return {'Loss': loss, 'l1_loss': w_1, 'mse_loss': w_2,
                'ssim_loss': w_s, 'ssim': ssim_m}

    def bce_mse(self, x, gt):
        a = torch.exp(gt)
        b = 1
        loss = a * x - (a + b) * torch.log(torch.exp(x) + 1)
        if self.reduction == 'none':
            return -loss
        if self.reduction == 'mean':
            return -torch.mean(loss)
        else:
            assert False, "there have no self.reduction choices"