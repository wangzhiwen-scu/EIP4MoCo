from torch import nn

def Sequential(cnn, norm, ac, bn=True):
    if bn:
        return nn.Sequential(cnn, norm, ac)
    else:
        return nn.Sequential(cnn, ac)

class ResBlock1(nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.01, bn=True):
        super(ResBlock1, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv6 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        
        out6 = self.conv6(out4) 
        output = self.ac6(out6 + x[:, 0:1, ...] + x[:, 1:2, ...]) 
        
        return output

class ResBlock2(nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.01, bn=True):
        super(ResBlock2, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)        
        self.conv6 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        
        out6 = self.conv6(out4) 
        
        output = self.ac6(out6 + x)
        return output

class ResBlock3(nn.Module):
    def __init__(self, in_ch, out_ch, negative_slope=0.01, bn=True):
        super(ResBlock3, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        
        
        
        
        self.conv6 = nn.Conv2d(32, out_ch, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        
        out6 = self.conv6(out4) 
        output = self.ac6(out6 + x[:, 0:1, ...] + x[:, 1:2, ...]) 
        
        return output

class LocalSemanticAttention(nn.Module): 
    def __init__(self):
        super(LocalSemanticAttention, self).__init__()
        print('using local structural loss')
        self.weight = [] 
        self.loss_l1 = nn.SmoothL1Loss()
        

    def forward(self, rec_result, gt, seg_result, seg_lbl):
        B, C, H, W = seg_lbl.shape
        
        
        
        
        
        
        
        
        
        
        
        total_loss = 0
        for i in range(0, C):
            gt_local = gt * seg_lbl[:, i:i+1, ...]
            rec_local = rec_result * seg_lbl[:, i:i+1, ...]
            
            total_loss += self.loss_l1(rec_local, gt_local) 
        return total_loss

def apply_crec_multiply_seg(rec_feature, seg_result):
    """rec_feature = (B, 1, H, W)
        seg_result = (B, C, H, W), C=N_CLASS
        

    Args:
        rec_feature (_type_): _description_
        seg_result (_type_): _description_

    Returns:
        _type_: _description_

    if dual net, get the abs of its results.
    """
    multi_channel_features = rec_feature * seg_result
    return multi_channel_features