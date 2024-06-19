import sys

import torch
from torch import nn

sys.path.append('.') 
from layers.mask_layer import (Hand_Tailed_Mask_Layer, Mask_CartesianLayer,
                               Mask_Fixed_CartesianLayer, Mask_Fixed_Layer,
                               Mask_Layer, Mask_Oneshot_Layer)
from model_backbones.recon_net import CSMRIUnet, ReconUnetforComb
from model_backbones.unet_cplx import ComplexUnet
from utils.train_utils import get_weights

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Comb_Net(nn.Module):
    def __init__(self, cartesian, nclasses, inputs_size, desired_sparsity, isfinetune, isfixseg, ckpt, isDC, args, sample_slope=10):
        super(Comb_Net, self).__init__()
        self.recon_net =  ReconUnetforComb(cartesian=cartesian, inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, isDC=isDC, isfinetune=isfinetune, ckpt=ckpt)


    def forward(self, x):
        out_recon, uifft, complex_abs, mask, fft, undersample = self.recon_net(x)
        return out_recon, uifft, complex_abs, mask, fft, undersample

class CSMTLNet(nn.Module):
    def __init__(self, args, nclasses, inputs_size, test=False):
        super(CSMTLNet, self).__init__()
        self.recon_net = CSMRIUnet(args.rate, args.mask, inputs_size)
    
    def forward(self, x):
        out_recon, uifft, complex_abs, mask, fft, undersample = self.recon_net(x)
        return out_recon, uifft, complex_abs, mask, fft, undersample


class CSMRIUnet_CPLX(nn.Module):
    def __init__(self, desired_sparsity, traj_type, inputs_size, isDC=False, in_ch=1, out_ch=1):
        super(CSMRIUnet_CPLX, self).__init__()
        self.mask_layer = Hand_Tailed_Mask_Layer(desired_sparsity, traj_type, inputs_size)
        self.unet_cplx = ComplexUnet(in_ch, out_ch)

    def forward(self, x):
        uifft, complex_abs, mask, fft, undersample = self.mask_layer(x)

        b,c,h,w = uifft.shape

        uifft = uifft.unsqueeze_(4)# b,cplx, h,w,1
        uifft = uifft.permute(0,4,2,3,1) # for complex last

        undersample = undersample.unsqueeze_(4)# b,cplx, h,w,1
        undersample = undersample.permute(0,4,2,3,1) # for complex last

        mask = mask.view(1,1,h,w,1).expand([b,1,h,w,2]) # mention: expand(origin) and repeat(copy)
        output = self.unet_cplx(uifft, undersample, mask)
        return output, uifft, complex_abs, mask, fft, undersample