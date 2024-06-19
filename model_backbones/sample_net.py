import torch
from torch import nn
from torch.nn import functional as F
import sys

sys.path.append('.') 
from model_backbones.unet import LUNet4Sampler

class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):

        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):

        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class Sampler2D(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1], fixed_input=False, roi_kspace=False):
        super(Sampler2D, self).__init__()
        
        in_chans = 5
        if roi_kspace:
            in_chans = 7
        out_chans = 1
        chans = 64
        num_pool_layers = 4
        drop_prob = 0

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        self.fixed_input = fixed_input
        if fixed_input:
            print('generate random input tensor')
            fixed_input_tensor = torch.randn(size=[1, 5, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        
        x = torch.cat([x, mask], dim=1)

        if self.fixed_input:
            x = self.fixed_input_tensor

        output = x 
        
        stack = []
        

        
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        out = self.conv2(output)

        out = F.softplus(out) 

        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1, 1, 1)

        
        new_mask = out * (1-mask)

        return new_mask


class KspaceLineConstrainedSampler(nn.Module):

    def __init__(self, in_chans, out_chans, clamp=100, with_uncertainty=False, fixed_input=False):
        super().__init__()

        if with_uncertainty:
            self.flatten_size = int((in_chans) **2 * 6)
        else:
            self.flatten_size = int((in_chans) **2 * 5)

        self.conv_last = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_chans)
        )

        self.clamp = clamp
        self.with_uncertainty = with_uncertainty
        self.fixed_input = fixed_input 

        if fixed_input:
            print("Generate random input tensor")
            fixed_input_tensor = torch.randn(size=[1, 5, in_chans, in_chans])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

        print("Finish Mask Initialization")

    def forward(self, kspace, mask, uncertainty_map=None):
        N, C, H, W = mask.shape

        if self.with_uncertainty:
            
            
            pass
        else:
            
            feat_map = torch.cat([kspace, mask], dim=1)

        if self.fixed_input:
            feat_map = self.fixed_input_tensor.repeat(N, 1, 1, 1).to(kspace.device) 

        out = feat_map.flatten(start_dim=1)

        out = self.conv_last(out)

        out = F.softplus(out) 

        
        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1)

        
        if out.shape[-1] == mask.shape[-1]:
            vertical_mask = (mask.sum(dim=-2) == H).float().reshape(N, -1)
            new_mask = out * (1-vertical_mask)
        else:
            
            vertical_mask = (mask.sum(dim=-2) == H).float().reshape(N, -1)
            horizontal_mask = (mask.transpose(-2, -1).sum(dim=-2)==W).float().reshape(N, -1)

            length = out.shape[1] // 2
            new_mask = torch.zeros_like(out)
            new_mask[:, :length] = out[:, :length] * (1-vertical_mask)
            new_mask[:, length:] = out[:, length:] * (1-horizontal_mask)

        
        return new_mask.reshape(N, 1, 1, -1) 


class LightSampler2D(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1], fixed_input=False):
        super(LightSampler2D, self).__init__()
        
        in_chans = 5
        out_chans = 1
        chans = 64
        num_pool_layers = 4
        drop_prob = 0

        self.unet = LUNet4Sampler(in_chans, out_chans)

        self.fixed_input = fixed_input
        if fixed_input:
            print('generate random input tensor')
            fixed_input_tensor = torch.randn(size=[1, 5, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, x, mask):
        
        x = torch.cat([x, mask], dim=1)
        if self.fixed_input:
            x = self.fixed_input_tensor
        output = x 
        out =self.unet(output)
        out = F.softplus(out) 
        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1, 1, 1)
        
        new_mask = out * (1-mask)
        return new_mask

class LightSampler1D(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1], fixed_input=False):
        super(LightSampler1D, self).__init__()
        print('sampler1D')
        in_chans = 5
        out_chans = 1
        chans = 64
        num_pool_layers = 4
        drop_prob = 0

        self.unet = LUNet4Sampler(in_chans, out_chans)

        self.fixed_input = fixed_input
        if fixed_input:
            print('generate random input tensor')
            fixed_input_tensor = torch.randn(size=[1, 5, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def forward(self, x, mask):
        
        N, C, H, W = mask.shape
        H = 1
        
        x = torch.cat([x, mask], dim=1)
        if self.fixed_input:
            x = self.fixed_input_tensor
        output = x 
        out =self.unet(output)
        out = torch.sum(out, dim=2, keepdims=True)
        out = F.softplus(out) 
        
        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1, 1, 1)
        
        
        vertical_mask = (mask.sum(dim=-2) == H).float().reshape(N, 1, 1, -1)
        
        new_mask = out * (1-vertical_mask)
        
        return new_mask

