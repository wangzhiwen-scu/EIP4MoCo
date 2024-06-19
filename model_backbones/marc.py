
import torch
from torch import nn
import torch.nn.functional as F

# file:///F:/library/Paper/motion_artifacts/ref_pdf/MARC.pdf


class StartConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(StartConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SingleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, input):
        return self.conv(input)


class MARC(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(MARC, self).__init__()
        self.startconv = StartConv(in_ch, 64)
        self.conv1 = SingleConv(64, 64)
        self.conv2 = SingleConv(64, 64)
        self.conv3 = SingleConv(64, 64)
        self.conv4 = SingleConv(64, 64)
        self.conv5 = SingleConv(64, 64)
        self.conv6 = SingleConv(64, 64)
        self.conv7 = SingleConv(64, 64)
        self.end = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        
        x = self.startconv(x)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.end(x)

        return x

    
if __name__ == "__main__":
    # global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # torch.backends.cudnn.benchmark = True
    net = MARC(1, 1).to(device, dtype=torch.float) # for complex one channel
    x = torch.rand(3, 1, 64, 64, dtype=torch.float).to(device)  # complex unet in torch-origin is usable.
    y = net(x)
    print(y.shape)