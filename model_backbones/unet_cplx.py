import torch
from torch import nn
import torch.nn.functional as F

import sys
sys.path.append('.') 
from utils.fourier_transform_utils import ifft2, fft2

class ComplexConv2d(nn.Module):
    # complex convolution
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()

        self.conv_real = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_imag = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):  # x.shape = (batch, channel, axis1, axis2, 2)
        real = self.conv_real(x[..., 0]) - self.conv_imag(x[..., 1])
        imag = self.conv_imag(x[..., 0]) + self.conv_real(x[..., 1])
        output = torch.stack((real, imag), dim=4)
        return output
    

class ComplexConvTranspose2d(nn.Module):
    # complex transpose convolution
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1, output_padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConvTranspose2d, self).__init__()

        self.conv_real = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_imag = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x, x1):  # x.shape = (batch, channel, axis1, axis2, 2)


        real = self.conv_real(x[..., 0]) - self.conv_imag(x[..., 1])

        diffY = x1.size()[2] - real.size()[2]
        diffX = x1.size()[3] - real.size()[3]

        real = F.pad(real, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        imag = self.conv_imag(x[..., 0]) + self.conv_real(x[..., 1])

        diffY = x1.size()[2] - imag.size()[2]
        diffX = x1.size()[3] - imag.size()[3]

        imag = F.pad(imag, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        output = torch.stack((real, imag), dim=4)       
        return output

    
class DataConsistency(nn.Module):
    # DC layer
    def __init__(self):
        super(DataConsistency, self).__init__()

    def forward(self, x, x_k, mask):
        # output = ifft2(x_k + fft2(x) * (1.0 - mask))
        output = torch.rand(size=x.size()).cuda()
        for i in range(len(x)):
            k_recon = fft2(x[i])* (1.0 - mask[i])
            output[i] = ifft2(x_k[i] + k_recon)
        return output


class ComplexUnet(nn.Module):

    def __init__(self, in_channels, out_channels):
        """_summary_

        Args:
            in_channels (_type_): _description_
            out_channels (_type_): _description_

            x = torch.rand(3, 1, 64, 64, 2, dtype=torch.float64).to(device)  # complex unet in torch-origin is usable.
            k = torch.rand(3, 1, 64, 64, 2, dtype=torch.float64).to(device)  # complex unet in torch-origin is usable.
            mask = torch.rand(3, 1, 64, 64, 2 ,dtype=torch.float64).to(device)  # complex unet in torch-origin is usable.
            
        """
        super(ComplexUnet, self).__init__()

        self.conv1 = ComplexConv2d(in_channel=in_channels, out_channel=64, kernel_size=3, stride=2)                 
        self.conv2 = ComplexConv2d(in_channel=64, out_channel=128, kernel_size=3, stride=2)                       
        self.conv3 = ComplexConv2d(in_channel=128, out_channel=256, kernel_size=3, stride=2)          
        self.conv4 = ComplexConv2d(in_channel=256, out_channel=512, kernel_size=3, stride=2)                       
        self.conv5 = ComplexConv2d(in_channel=512, out_channel=1024, kernel_size=3, stride=2)                    

        self.deconv1 = ComplexConvTranspose2d(in_channel=1024, out_channel=512, kernel_size=3, stride=2) # output_padding=1    
        self.deconv2 = ComplexConvTranspose2d(in_channel=512, out_channel=256, kernel_size=3, stride=2)            
        self.deconv3 = ComplexConvTranspose2d(in_channel=256, out_channel=128, kernel_size=3, stride=2)          
        self.deconv4 = ComplexConvTranspose2d(in_channel=128, out_channel=64, kernel_size=3, stride=2)             
        self.deconv5 = ComplexConvTranspose2d(in_channel=64, out_channel=out_channels, kernel_size=3, stride=2, output_padding=1)  

        self.relu = nn.ReLU()
        self.dc = DataConsistency()

    def forward(self, x, x_k, mask):
        # UNet encoder
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        conv4 = self.relu(self.conv4(conv3))
        conv5 = self.relu(self.conv5(conv4))
        # UNet decoder
        deconv1 = self.relu(self.deconv1(conv5, conv4)) + conv4 # 1.512,14,14,2
        deconv2 = self.relu(self.deconv2(deconv1, conv3)) + conv3
        deconv3 = self.relu(self.deconv3(deconv2, conv2)) + conv2
        deconv4 = self.relu(self.deconv4(deconv3, conv1)) + conv1
        deconv5 = self.relu(self.deconv5(deconv4, x))
        # DC layer
        output = self.dc(deconv5, x_k, mask)

        return output
    
if __name__ == "__main__":
    # global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.backends.cudnn.benchmark = True
    net = ComplexUnet(1, 1).to(device, dtype=torch.float64) # for complex one channel
    x = torch.rand(3, 1, 240, 240, 2, dtype=torch.float64).to(device)  # complex unet in torch-origin is usable.
    k = torch.rand(3, 1, 240, 240, 2, dtype=torch.float64).to(device)  # complex unet in torch-origin is usable.
    b,c,h,w,_ = k.shape
    mask = torch.rand(240, 240, dtype=torch.float64).to(device)  # complex unet in torch-origin is usable.
    mask = mask.view(1,1,h,w,1).expand([b,c,h,w,2])
    y = net(x, k, mask)

    kk = 0