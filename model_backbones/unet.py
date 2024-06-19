
import torch
from torch import nn
import torch.nn.functional as F

# class DoubleConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.BatchNorm2d(out_ch),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.LeakyReLU(inplace=True),
#             nn.BatchNorm2d(out_ch),
#         )
    
#     def forward(self, input):
#         return self.conv(input)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, input):
        return self.conv(input)

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


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def down_sample_v0(in_channel, channels):
    return nn.Sequential(
        nn.Conv2d(in_channel, channels, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
    )

def down_sample_v1(in_channel, channels):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, channels, kernel_size=1, padding=0, stride=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
    )

def up_sample(in_channle):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channle, in_channle//2, kernel_size=2, stride=2),
    )
 
def up_conv_v0(in_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel//2, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU(),
        nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU()
    )
def up_conv_v1(in_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, padding=0, stride=1),
        nn.BatchNorm2d(in_channel // 2),
        nn.ReLU(),
        nn.Conv2d(in_channel // 2, in_channel//2, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU(),
        nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, padding=0, stride=1),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU()
    )

def down_sample_v2(in_channel, channels):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, channels, kernel_size=1, padding=0, stride=1),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1, groups=4),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
    )

def up_conv_v2(in_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, padding=0, stride=1, groups=4),
        nn.BatchNorm2d(in_channel // 2),
        nn.ReLU(),
        nn.Conv2d(in_channel // 2, in_channel//2, kernel_size=3, padding=0, stride=1, groups=4),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU(),
        nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, padding=0, stride=1, groups=4),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU()
    )

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depth_wise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels, (1, 1), 1, 0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.depth_wise_conv(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.point_wise_conv(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x

def down_sample(in_channel, channels): 
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, groups=in_channel),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, channels, kernel_size=1, padding=0, stride=1, groups=in_channel),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
        nn.Conv2d(channels, channels, kernel_size=3, padding=0, stride=1, groups=in_channel),
        nn.BatchNorm2d(channels),
        nn.ReLU(),
    )

def up_conv(in_channel): 
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel // 2, kernel_size=1, padding=0, stride=1),
        nn.BatchNorm2d(in_channel // 2),
        nn.ReLU(),
        nn.Conv2d(in_channel // 2, in_channel//2, kernel_size=3, padding=0, stride=1, groups=in_channel),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU(),
        nn.Conv2d(in_channel//2, in_channel//2, kernel_size=3, padding=0, stride=1, groups=in_channel),
        nn.BatchNorm2d(in_channel//2),
        nn.ReLU()
    )

def cat(x1, x2):
    diff = x1.size()[2] - x2.size()[2]
    x2 = F.pad(x2, [diff // 2, diff - diff // 2,
                    diff // 2, diff - diff // 2])
    x2 = torch.cat([x1, x2], dim=1)
    return x2
 
class Unet(nn.Module):
    def __init__(self, in_channel=1, out_channel=1):
        super(Unet, self).__init__()
 

        self.down_sample1 = down_sample(in_channel, 64)
        self.down_sample2 = down_sample(64, 128)
        self.down_sample3 = down_sample(128, 256)
        self.down_sample4 = down_sample(256, 512)
        self.down_sample5 = down_sample(512, 1024)
 
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
 
        self.up_sample1 = up_sample(1024)
        self.up_sample2 = up_sample(512)
        self.up_sample3 = up_sample(256)
        self.up_sample4 = up_sample(128)
 
        self.up_conv1 = up_conv(1024)
        self.up_conv2 = up_conv(512)
        self.up_conv3 = up_conv(256)
        self.up_conv4 = up_conv(128)
 
        self.out_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=0, stride=1)
        )
 
    def forward(self, x):
        x1 = self.down_sample1(x)
        p1 = self.pooling(x1)
        x2 = self.down_sample2(p1)
        p2 = self.pooling(x2)
        x3 = self.down_sample3(p2)
        p3 = self.pooling(x3)
        x4 = self.down_sample4(p3)
        p4 = self.pooling(x4)
        x5 = self.down_sample5(p4)
 
        
        x6 = self.up_sample1(x5)
        x6 = cat(x4, x6)
 
        x7 = self.up_conv1(x6)
        x7 = self.up_sample2(x7)
        x7 = cat(x3, x7)
 
        x8 = self.up_conv2(x7)
        x8 = self.up_sample3(x8)
        x8 = cat(x2, x8)
 
        x9 = self.up_conv3(x8)
        x9 = self.up_sample4(x9)
        x9 = cat(x1, x9)
 
        output = self.out_conv(x9)
        return output


class depthwise_pointwise_conv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, padding, channels_per_seg=128):
        super(depthwise_pointwise_conv, self).__init__()
        if in_ch==3 and channels_per_seg!=1:
            C=3
        else:
            C=min(in_ch, channels_per_seg)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch//C),
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, groups=1),

        )
    def forward(self, x):
        x = self.conv(x)
        return x

class LightDoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            depthwise_pointwise_conv(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            depthwise_pointwise_conv(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            LightDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = LightDoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = LightDoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LUNet(nn.Module):
    """Lighter_UNet

    Args:
        nn (_type_): _description_
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(LUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = LightDoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x + x0
        return x

class LUNet4Sampler(nn.Module):
    """Lighter_UNet for LUNet4Sampler

    Args:
        nn (_type_): _description_
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(LUNet4Sampler, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
if __name__ == "__main__":
    # global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.backends.cudnn.benchmark = True
    net = ReconUnet(1, 1).to(device, dtype=torch.cfloat) # for complex one channel
    x = torch.rand(3, 1, 64, 64, dtype=torch.cfloat).to(device)  # complex unet in torch-origin is usable.
    y = net(x)
