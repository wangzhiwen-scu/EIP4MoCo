from torch import nn

def Sequential(cnn, norm, ac, bn=True):
    if bn:
        return nn.Sequential(cnn, norm, ac)
    else:
        return nn.Sequential(cnn, ac)
        
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

class ReconstructionForwardUnit(nn.Module):
    def __init__(self, bn):
        super(ReconstructionForwardUnit, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv2 = Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv3 = Sequential(
            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv4 = Sequential(
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv5 = Sequential(
            nn.Conv2d(256, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv6 = Sequential(
            nn.Conv2d(128, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv7 = Sequential(
            nn.Conv2d(64, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True), bn=bn)
        self.conv8 = nn.Conv2d(32, 1, 3, padding=1)
        self.ac8 = nn.LeakyReLU(inplace=True)


    def forward(self, *input):
        x, u_x = input
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        output = self.ac8(x8 + u_x)
        return output

class FeatureExtractor(nn.Module):
    def __init__(self, bn):
        super(FeatureExtractor, self).__init__()
        ############################################################
        # self.kspace_extractor = FeatureResidualUnit()
        # self.image_extractor = FeatureResidualUnit()

        ###########################################################
        self.kspace_extractor = FeatureForwardUnit(bn=bn)
        self.image_extractor = FeatureForwardUnit(bn=bn)

        ############################################################

        initialize_weights(self)

    def forward(self, *input):
        k, img = input
        k_feature = self.kspace_extractor(k)
        img_feature = self.image_extractor(img)

        return k_feature, img_feature

class FeatureForwardUnit(nn.Module):
    def __init__(self, negative_slope=0.01, bn=True):
        super(FeatureForwardUnit, self).__init__()
        self.conv1 = Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
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
        # self.conv5 = Sequential(
        #     nn.Conv2d(32, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(negative_slope=negative_slope), bn=bn)
        self.conv6 = nn.Conv2d(32, 2, 3, padding=1)
        self.ac6 = nn.LeakyReLU(negative_slope=negative_slope)
    
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        # out5 = self.conv5(out4)
        out6 = self.conv6(out4)
        output = self.ac6(out6 + x)

        return output