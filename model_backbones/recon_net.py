import sys
import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('.') 
from layers.mask_layer import Mask_Layer, Mask_Fixed_Layer, Mask_CartesianLayer, Mask_Fixed_CartesianLayer, Hand_Tailed_Mask_Layer, Mask_Oneshot_Layer, Hand_Tailed_Mask_Layer4SSL
from layers.conv_layer import DoubleConv, DuDoRnetDC, DC
from layers.att_unet_module import conv_block, up_conv, Attention_block
from layers.resblock import ResBlock1, ResBlock2

def unsampling_padding(x, x1):
    diffY = x1.size()[2] - x.size()[2]
    diffX = x1.size()[3] - x.size()[3]
    x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    return x

class ReconUnetforComb(nn.Module):
    def __init__(self, cartesian, inputs_size, desired_sparsity, isDC, sample_slope=10, isfinetune=False, ckpt=None, in_ch=1, out_ch=1):
        super(ReconUnetforComb, self).__init__()
        if not cartesian:
            if isfinetune:
                self.mask_layer = Mask_Fixed_Layer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, ckpt=ckpt)
                
                
                for param in self.mask_layer.parameters():
                    param.requires_grad = False
            else:
                self.mask_layer = Mask_Layer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity)
        elif cartesian:
            if isfinetune:
                self.mask_layer = Mask_Fixed_CartesianLayer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, ckpt=ckpt)
                
                
                for param in self.mask_layer.parameters():
                    param.requires_grad = False
            else:
                self.mask_layer = Mask_CartesianLayer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity)            
        
        
        self.desired_sparsity = desired_sparsity
        self.isDC = isDC
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)
        
        self.dc = DuDoRnetDC() 
        

    
    def forward(self, x):

        uifft, complex_abs, mask, fft, undersample = self.mask_layer(x)
        c1=self.conv1(complex_abs)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)

        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        out2 = c10 + complex_abs
        
            
        
            
        
            
            

        return out2, uifft, complex_abs, mask, fft, undersample



class CSMRIUnet(nn.Module):
    def __init__(self, desired_sparsity, traj_type, inputs_size, isDC=False, in_ch=2, out_ch=1):
        super(CSMRIUnet, self).__init__()

        
        self.mask_layer = Hand_Tailed_Mask_Layer(desired_sparsity, traj_type, inputs_size)
        self.isDC = isDC
        self.conv1 = DoubleConv(2, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)
        if isDC:
            self.dc = DC()
            print('\nusing DC Layer.')
    
    def forward(self, x):
        uifft, complex_abs, mask, fft, undersample = self.mask_layer(x)
                
        
        c1=self.conv1(uifft)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        out = c10 + complex_abs
        if self.isDC:
            out2 = self.dc(out, mask, True)
        
        return out, uifft, complex_abs, mask, fft, undersample

class CSMRIUnet_CH2(nn.Module):
    def __init__(self, desired_sparsity, traj_type, inputs_size, isDC=False, in_ch=2, out_ch=2):
        super(CSMRIUnet_CH2, self).__init__()

        
        self.mask_layer = Hand_Tailed_Mask_Layer(desired_sparsity, traj_type, inputs_size)
        self.isDC = isDC
        self.conv1 = DoubleConv(2, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)
        if isDC:
            self.dc = DC()
            print('\nusing DC Layer.')
    
    def forward(self, x):
        uifft, complex_abs, mask, fft, undersample = self.mask_layer(x)
                
        
        c1=self.conv1(uifft)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        out = c10 + uifft
        if self.isDC:
            out2 = self.dc(out, mask, True)
        
        return out, uifft, complex_abs, mask, fft, undersample

class Unet(nn.Module):
    def __init__(self, isDC=False, in_ch=1, out_ch=1):
        super(Unet, self).__init__()
       
        self.isDC = isDC
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)

        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)
        if isDC:
            self.dc = DC()
            print('\nusing DC Layer.')
    
    def forward(self, x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        # up_6 = unsampling_padding(up_6, c4)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        # up_7 = unsampling_padding(up_7, c3)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        # up_8 = unsampling_padding(up_8, c2)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        # up_9 = unsampling_padding(up_9, c1)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        if x.shape[1] == 3:
            out = c10 + x[:,1:2, :,:]
        else:
            out = c10 + x
        # out = c10
        if self.isDC:
            out2 = self.dc(out, True)
        return out

class CSLUnet(nn.Module):
    def __init__(self, cartesian, inputs_size, desired_sparsity, isDC, sample_slope=10, isfinetune=False, ckpt=None, in_ch=2, out_ch=1):
        super(CSLUnet, self).__init__()
        if not cartesian:
            if isfinetune:
                self.mask_layer = Mask_Fixed_Layer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, ckpt=ckpt)
                
                
                for param in self.mask_layer.parameters():
                    param.requires_grad = False
            else:
                self.mask_layer = Mask_Layer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity)
        elif cartesian:
            if isfinetune:
                self.mask_layer = Mask_Fixed_CartesianLayer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, ckpt=ckpt)
                
                
                for param in self.mask_layer.parameters():
                    param.requires_grad = False
            else:
                self.mask_layer = Mask_CartesianLayer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity)            
        
        
        self.conv1 = DoubleConv(1, 64)  
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)

    
    def forward(self, x):

        uifft, complex_abs, mask, fft, undersample = self.mask_layer(x)
        
        
        
        c1=self.conv1(complex_abs)  
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        up_6= self.up6(c5)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = c10 + complex_abs
        return out, uifft, complex_abs, mask, fft, undersample

class CSLOneShotUnet(nn.Module):
    def __init__(self, cartesian, inputs_size, desired_sparsity, isDC, sample_slope=10, isfinetune=False, ckpt=None, in_ch=2, out_ch=1):
        super(CSLOneShotUnet, self).__init__()
        if not cartesian:
            self.mask_layer = Mask_Oneshot_Layer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity)

        elif cartesian:
            if isfinetune:
                self.mask_layer = Mask_Fixed_CartesianLayer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity, ckpt=ckpt)
                
                
                for param in self.mask_layer.parameters():
                    param.requires_grad = False
            else:
                self.mask_layer = Mask_CartesianLayer(inputs_size=inputs_size, sample_slope=sample_slope, desired_sparsity=desired_sparsity)            
        
        
        self.conv1 = DoubleConv(1, 64)  
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)

    
    def forward(self, x):

        uifft, complex_abs, mask, fft, undersample = self.mask_layer(x)
        
        
        
        c1=self.conv1(complex_abs)  
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        up_6= self.up6(c5)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = c10 + complex_abs
        return out, uifft, complex_abs, mask, fft, undersample

class ReconUnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(ReconUnet, self).__init__()
        print('using classical unet')
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)

        

    
    def forward(self, complex_abs):

        c1=self.conv1(complex_abs)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        up_6= self.up6(c5)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        out = c10 + complex_abs
        return out

class ReconMidLossUnet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super(ReconMidLossUnet, self).__init__()
        print('using classical unet')
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.AvgPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.AvgPool2d(2)
        self.conv5 = DoubleConv(512, 1024)  
        
        self.up6 = nn.Upsample(scale_factor=2)
        self.conv6 = DoubleConv(1024+512, 512)  
        
        self.up7 = nn.Upsample(scale_factor=2) 
        self.conv7 = DoubleConv(512+256, 256)  
        
        self.up8 = nn.Upsample(scale_factor=2)
        self.conv8 = DoubleConv(256+128, 128)  
        
        self.up9 = nn.Upsample(scale_factor=2)
        self.conv9 = DoubleConv(128+64, 64)  
        self.conv10 = nn.Conv2d(64, out_ch, kernel_size=1)

        

    
    def forward(self, complex_abs):

        c1=self.conv1(complex_abs)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)

        up_6= self.up6(c5)
        merge6 = torch.cat([c4, up_6], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([c3, up_7], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([c2, up_8], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([c1, up_9], dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        
        out = c10 + complex_abs
        return out, c9, c8, c7, c6, c5

class TinyAttU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(TinyAttU_Net, self).__init__()

        print('using TinyAttU_Net')
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)


        d4 = self.Up4(e4)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        rec = self.Conv(d2)
        # rec = rec + x


        return rec

class SARNet(nn.Module):

    def __init__(self, img_ch, output_ch, n_cls):
        super(SARNet, self).__init__()
        self.n_cls = n_cls
        self.resblock1 = ResBlock1(in_ch=2, out_ch=32)  
        self.resblock2 = self.build_resblock2(img_ch=32, output_ch=1, n_cls=n_cls)
        

    def build_resblock2(self, img_ch, output_ch, n_cls):
        blocklist= []
        
        for i in range(1, n_cls):
            blocklist.append(ResBlock2(in_ch=img_ch, out_ch=output_ch))
        return nn.ModuleList(blocklist)

    def apply_crec_multiply_seg(self, rec_feature, seg_result):
        """rec_feature = (B, 32, H, W)
            seg_result = (B, C, H, W), C=N_CLASS
        if dual net, get the abs of its results.
        """
        features_semantic = []
        B,C,H,W = seg_result.shape
        for i in range(C):
            
            features_semantic.append(rec_feature * seg_result[:, i:i+1, ...])

        
        return features_semantic

    def splitted_semantic_resblock(self, semantic_features):
        
        
        
        fine_rec =0
        for i, ith_resblock in enumerate(self.resblock2):
            
            fine_rec += ith_resblock(semantic_features[i])
        fine_rec = fine_rec.sum(1, keepdim=True)
        return fine_rec

    def forward(self, uifft, crec_abs, seg_result):
        fusion_cat = torch.cat([uifft, crec_abs], dim=1)  
        features_before = self.resblock1(fusion_cat)  
        semantic_features = self.apply_crec_multiply_seg(features_before, seg_result) 
        
        fine_rec = self.splitted_semantic_resblock(semantic_features)  
        
        

        return fine_rec


class REDCNN(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = 96

        self.l1 = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, 5, padding=2, stride=1),
            nn.ReLU(),
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
            nn.ReLU()
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
            nn.ReLU()
        )

        self.l5 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
            nn.ReLU()
        )

        self.l6 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
        )

        self.l7 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
        )

        self.l8 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
        )

        self.l9 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, 5, padding=2, stride=1),
        )

        self.l10 = nn.Sequential(
            nn.Conv2d(self.out_ch, self.in_ch, 5, padding=2, stride=1),
        )

    def forward(self, x):
        conv1 = self.l1(x)  
        conv2 = self.l2(conv1)  
        conv3 = self.l3(conv2)  
        conv4 = self.l4(conv3)  
        conv5 = self.l5(conv4)  

        deconv6 = self.l6(conv5)  
        deconv6 += conv4
        deconv6 = F.relu(deconv6)

        deconv7 = self.l7(deconv6)  
        deconv7 = F.relu(deconv7)

        deconv8 = self.l8(deconv7)
        deconv8 += conv2
        deconv8 = F.relu(deconv8)
        
        deconv9 = self.l9(deconv8)
        deconv9 = F.relu(deconv9)
        
        deconv10 = self.l10(deconv9)
        deconv10 += x
        deconv10 = deconv10

        return deconv10
    
class MedCycleU_Net(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(MedCycleU_Net, self).__init__()

        print('using MedCycleU_Net')
        # input.shape = (B,1, 240,240)
        self.Conv1 = nn.Conv2d(img_ch, 64, kernel_size=7, stride=1, padding=1, bias=True) # (B,1, 240,240) -> (B,64, 240,240)
        self.Conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True) # (B,64, 240,240) -> (B,128, 120, 120)

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        
        # self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        # e2 = self.Conv2(e2)
        # e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        d4 = self.Up4(e4)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        rec = self.Conv(d2)
        rec = rec + x

        return rec
