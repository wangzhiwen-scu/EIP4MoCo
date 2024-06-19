import torch
import numpy as np
from torch import nn
# train part
import torch.nn.functional as F
# from pytorch_msssim import ssim
from torch import nn
from layers.dota_mri_layer import fft2

class Metrics():
    def __init__(self):
        self.psnrs = []
        self.ssims = []
        # self.dices = []
        # self.ious = []
        self.mean_psnr = 0
        self.mean_ssim = 0
        # self.dices_mean_var = ''
        # self.ious_mean_var = ''
        
    def get_metrics(self, psnr, ssim):
        self.psnrs.append(round(psnr, 2))
        self.ssims.append(round(ssim, 2))
        # self.dices.append(round(dice, 2))
        # self.ious.append((iou, 2))
        self.mean_psnr = round(np.mean(self.psnrs), 2)
        self.mean_ssim = 100*round(np.mean(self.ssims), 3)

        # self.psnrs_mean_var = str(self.mean_psnr) + '+-' +  str(round(np.std(self.psnrs), 2))
        # self.ssims_mean_var = str(self.mean_ssim) + '+-' +  str(round(100*np.std(self.ssims), 2))
        # self.dices_mean_var = str(round(100*np.mean(self.dices), 2)) + '+-' +  str(round(100*np.std(self.dices), 2))
        # self.ious_mean_var = str(round(100*np.mean(self.ious), 2)) + '+-' +  str(round(100*np.std(self.dices), 2))

def dice_coef(output, target):
    """Dice co.
    """
    # def dice_coef(output, target):#output为预测结果 target为真实结果
    # smooth = 1e-5 #防止0除
    smooth = 0
    if torch.is_tensor(output):
        output = output.detach().data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.detach().data.cpu().numpy()
    output[output > 0.5] = 1.0 #!
    # if np.mean(output) < 0.001: #!
    #     output[output ==1] = 0
    # target_ = target > 0.5
    intersection = (output * target).sum()
    # intersection = (output_ * target_).sum()
    try:
        dice =  (2. * intersection + smooth) / \
            (output.sum() + target.sum() + smooth)
    except ZeroDivisionError:
        dice = np.nan
    # if np.mean(output) < 0.001: #!
        # dice = 0
    return dice

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)

        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class LocalStructuralLoss(nn.Module): # Adaptive local structural loss Exploiting Semantics for Face Image Deblurring
    def __init__(self):
        super(LocalStructuralLoss, self).__init__()
        self.weight = [] # tiny component may not be well recon,  when weight_all =1
        self.loss = nn.SmoothL1Loss()

    def forward(self, rec_result, gt, seg_result, seg_lbl):
        # B, C, H, W = seg_result.shape
        seg_sigmoid = torch.sigmoid(seg_result)
        # seg_sigmoid = seg_sigmoid[:,0:3,...]
        # seg_lbl = seg_lbl[:,0:3,...]
        # seg_binary = torch.zeros_like(seg_result)
        # gt_local = torch.zeros_like(seg_result)
        # rec_local = torch.zeros_like(seg_result)
        # seg_binary[seg_sigmoid >= 0.5] = 1
        # seg_binary[seg_sigmoid < 0.5] = 0        
        # for c in range(0, C):
        
        gt_local = gt * seg_lbl
        rec_sigmoid = rec_result * seg_sigmoid

        rec_gt = rec_result * seg_lbl

        full_sigmoid = gt * seg_sigmoid

        # rec_local = rec_result * seg_lbl

        total_loss = self.loss(rec_sigmoid, gt_local) + 1*self.loss(rec_gt, gt_local) + 1*self.loss(full_sigmoid, gt_local)

        return total_loss

def content_loss(input, target):
    # target = target.detach()
    loss = F.mse_loss(input, target)
    return loss

class ContentLoss(nn.Module): # Adaptive local structural loss Exploiting Semantics for Face Image Deblurring
    def __init__(self):
        super(ContentLoss, self).__init__()
        print('using local structural loss')
        self.weight = [] # tiny component may not be well recon,  when weight_all =1
        self.loss_l1 = nn.SmoothL1Loss()
        
    def forward(self, features_u, features_f):
        total_loss = 0
        for feat_u, feat_f in zip(features_u, features_f):
            # total_loss += self.loss_perceptual(rec_local, gt_local) # inputs, target.
            total_loss += self.loss_l1(feat_u, feat_f) # inputs, target.
        return total_loss

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        
    def forward(self, input, target_feature):
        G = gram_matrix(input)
        target = gram_matrix(target_feature).detach()

        self.loss = F.mse_loss(G, target)
        return input

def  get_style_losses_and_content_losses(input, gt):
    # _c9, _c8, _c7, _c6, _c5 = 

    pass



class dcloss(nn.Module):
    def __init__(self):
        super(dcloss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, img_recon, k_input, mask):
        k_recon = fft2(img_recon)
        loss = self.loss(k_recon * mask, k_input)

        return loss
    
class imgloss(nn.Module):
    def __init__(self):
        super(imgloss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, img_recon, img_label):
        loss = self.loss(img_recon, img_label)

        return loss



# def batch_mri_ssim_loss(features_rec, features_seg):
#     # features should be (b,1,h,w)
#     # https://github.com/VainF/pytorch-msssim
#     b = features_rec.shape[0]

#     total_loss = 0
#     for i in range(b):
#         # total_loss += self.loss_perceptual(rec_local, gt_local) # inputs, target.
#         batch_max = torch.max(features_rec[i])
#         ssim_loss = 1 - ssim(features_rec[i:i+1], features_seg[i:i+1], data_range=batch_max, size_average=True) # inputs, target.
#         total_loss += ssim_loss
#     return total_loss
