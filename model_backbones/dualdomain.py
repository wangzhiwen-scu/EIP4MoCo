import sys
import torch
import torch.nn as nn
sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from layers.dualdomain_layer import ReconstructionForwardUnit, FeatureExtractor

ReconstructionUnit = ReconstructionForwardUnit

def fft(input):
    # (N, 2, W, H) -> (N, W, H, 2)
    # print(type(input))
    input = input.permute(0, 2, 3, 1)
    input = torch.fft(input, 2, normalized=False) # !

    # (N, W, H, 2) -> (N, 2, W, H)
    input = input.permute(0, 3, 1, 2)
    return input

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2, normalized=False) # !

    # (N, D, W, H, 2) -> (N, 2, D, W, H)
    input = input.permute(0, 3, 1, 2)

    return input


class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, rec, u_k, mask, is_img=False):
        if is_img:
            rec = fft(rec)
        result = mask * (rec * self.w / (1 + self.w) + u_k * 1 / (self.w + 1)) # weighted the undersampling and reconstruction
        result = result + (1 - mask) * rec # non-sampling point

        if is_img:
            result = ifft(result)
        
        return result



class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, x1, x2):
        return x1 * 1 / (1 + self.w) + x2 * self.w / (self.w + 1)



class MRIReconstruction(nn.Module):
    def __init__(self, mask, w, bn):
        super(MRIReconstruction, self).__init__()
        self.cnn1 = FeatureExtractor(bn=bn)
        self.dc11 = DC()
        self.dc12 = DC()
        self.fusion11 = Fusion()
        self.fusion12 = Fusion()

        self.cnn2 = FeatureExtractor(bn=bn)
        self.dc21 = DC()
        self.dc22 = DC()
        self.fusion21 = Fusion()
        self.fusion22 = Fusion()


        self.cnn3 = FeatureExtractor(bn=bn)
        self.dc31 = DC()
        self.dc32 = DC()
        self.fusion31 = Fusion()
        self.fusion32 = Fusion()


        self.cnn4 = FeatureExtractor(bn=bn)
        self.dc41 = DC()
        self.dc42 = DC()
        self.fusion41 = Fusion()
        self.fusion42 = Fusion()

        self.cnn5 = FeatureExtractor(bn=bn)
        self.dc51 = DC()
        self.dc52 = DC()
        self.fusion51 = Fusion()
        # self.fusion52 = Fusion()

        self.mask = mask
        self.w = w

    def forward(self, *input):
        ############################## First Stage ######################################
        # resstore feature from raw data
        k_x_1 = input[0]
        img_x_1 = input[1]

    # def forward(self, input): # for flops counting.
    #     k_x_1 = input
    #     img_x_1 = input

        u_k = k_x_1

        k_fea_1, img_fea_1 = self.cnn1(*(k_x_1, img_x_1))

        rec_k_1 = self.dc11(k_fea_1, u_k, self.mask)
        rec_img_1 = self.dc12(img_fea_1, u_k, self.mask, True)

        k_to_img_1 = ifft(rec_k_1)  # convert the restored kspace to spatial domain
        img_to_k_1 = fft(rec_img_1) # convert the restored image to frequency domain


        ################################ Second Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_2 = self.fusion11(rec_k_1, img_to_k_1)
        img_x_2 = self.fusion12(rec_img_1, k_to_img_1)

        k_fea_2, img_fea_2 = self.cnn2(*(k_x_2, img_x_2))

        rec_k_2 = self.dc21(k_fea_2, u_k, self.mask)
        rec_img_2 = self.dc22(img_fea_2, u_k, self.mask, True)

        k_to_img_2 = ifft(rec_k_2)  # convert the restored kspace to spatial domain
        img_to_k_2 = fft(rec_img_2) # convert the restored image to frequency domain


        ################################ Third Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_3 = self.fusion21(rec_k_2, img_to_k_2)
        img_x_3 = self.fusion22(rec_img_2, k_to_img_2)

        k_fea_3, img_fea_3 = self.cnn3(*(k_x_3, img_x_3))

        rec_k_3 = self.dc31(k_fea_3, u_k, self.mask)
        rec_img_3 = self.dc32(img_fea_3, u_k, self.mask, True)

        k_to_img_3 = ifft(rec_k_3)  # convert the restored kspace to spatial domain
        img_to_k_3 = fft(rec_img_3) # convert the restored image to frequency domain


        ################################ Forth Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_4 = self.fusion31(rec_k_3, img_to_k_3)
        img_x_4 = self.fusion32(rec_img_3, k_to_img_3)

        k_fea_4, img_fea_4 = self.cnn4(*(k_x_4, img_x_4))

        rec_k_4 = self.dc41(k_fea_4, u_k, self.mask)
        rec_img_4 = self.dc42(img_fea_4, u_k, self.mask,  True)

        k_to_img_4 = ifft(rec_k_4)  # convert the restored kspace to spatial domain
        img_to_k_4 = fft(rec_img_4) # convert the restored image to frequency domain


        ################################ Third Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_5 = self.fusion41(rec_k_4, img_to_k_4)
        img_x_5 = self.fusion42(rec_img_4, k_to_img_4)

        k_fea_5, img_fea_5 = self.cnn5(*(k_x_5, img_x_5))

        rec_k_5 = self.dc51(k_fea_5, u_k, self.mask)
        rec_img_5 = self.dc52(img_fea_5, u_k, self.mask, True)

        k_to_img_5 = ifft(rec_k_5)  # convert the restored kspace to spatial domain

        out = self.fusion51(rec_img_5, k_to_img_5)

        return out

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, x1, x2):
        return x1 * 1 / (1 + self.w) + x2 * self.w / (self.w + 1)

def torch_binarize(k_data, desired_sparsity):
    batch, channel, x_h, x_w = k_data.shape
    tk = int(desired_sparsity*x_h*x_w)+1
    k_data = k_data.reshape(batch, channel, x_h*x_w, 1)
    values, indices = torch.topk(k_data, tk, dim=2)
    k_data_binary =  (k_data >= torch.min(values))
    k_data_binary = k_data_binary.reshape(batch, channel, x_h, x_w).float()
    return k_data_binary

class DuDoRnetDC(nn.Module):
    def __init__(self):
        super(DuDoRnetDC, self).__init__()
        self.v = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        print('\nusing DuDoRnetDC Layer.')
    
    def forward(self, rec, u_k, mask, desired_sparsity):
        """ https://github.com/bbbbbbzhou/DuDoRNet
        k    - input in k-space
        u_k   - initially sampled elements in k-space,  k0
        mask - corresponding nonzero location
        v    - noise_lvl
        """
        mask = torch_binarize(mask, desired_sparsity)
        rec_k = fft(rec)
        out = (1 - mask) * rec_k + mask * (rec_k + self.v * u_k) / (1 + self.v)
        result = ifft(out)
        return result

class DualDoRecNet(nn.Module):
    def __init__(self, w=0.2, bn=False):
        super(DualDoRecNet, self).__init__()
        self.cnn1 = FeatureExtractor(bn=bn)
        self.dc11 = DC()
        self.dc12 = DC()
        self.fusion11 = Fusion()
        self.fusion12 = Fusion()

        self.cnn2 = FeatureExtractor(bn=bn)
        self.dc21 = DC()
        self.dc22 = DC()
        self.fusion21 = Fusion()
        self.fusion22 = Fusion()


        self.cnn3 = FeatureExtractor(bn=bn)
        self.dc31 = DC()
        self.dc32 = DC()
        self.fusion31 = Fusion()
        self.fusion32 = Fusion()


        self.cnn4 = FeatureExtractor(bn=bn)
        self.dc41 = DC()
        self.dc42 = DC()
        self.fusion41 = Fusion()
        self.fusion42 = Fusion()

        self.cnn5 = FeatureExtractor(bn=bn)
        self.dc51 = DC()
        self.dc52 = DC()
        self.fusion51 = Fusion()
        # self.fusion52 = Fusion()

        self.w = w

    def forward(self, *input):
        ############################## First Stage ######################################
        # resstore feature from raw data
        k_x_1 = input[0]
        img_x_1 = input[1]
        learned_mask = input[2]
        # learned_mask = torch_binarize(learned_mask, self.desired_sparsity)
        u_k = k_x_1

        k_fea_1, img_fea_1 = self.cnn1(*(k_x_1, img_x_1))

        rec_k_1 = self.dc11(k_fea_1, u_k, learned_mask)
        rec_img_1 = self.dc12(img_fea_1, u_k, learned_mask, True)

        k_to_img_1 = ifft(rec_k_1)  # convert the restored kspace to spatial domain
        img_to_k_1 = fft(rec_img_1) # convert the restored image to frequency domain


        ################################ Second Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_2 = self.fusion11(rec_k_1, img_to_k_1)
        img_x_2 = self.fusion12(rec_img_1, k_to_img_1)

        k_fea_2, img_fea_2 = self.cnn2(*(k_x_2, img_x_2))

        rec_k_2 = self.dc21(k_fea_2, u_k, learned_mask)
        rec_img_2 = self.dc22(img_fea_2, u_k, learned_mask, True)

        k_to_img_2 = ifft(rec_k_2)  # convert the restored kspace to spatial domain
        img_to_k_2 = fft(rec_img_2) # convert the restored image to frequency domain


        ################################ Third Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_3 = self.fusion21(rec_k_2, img_to_k_2)
        img_x_3 = self.fusion22(rec_img_2, k_to_img_2)

        k_fea_3, img_fea_3 = self.cnn3(*(k_x_3, img_x_3))

        rec_k_3 = self.dc31(k_fea_3, u_k, learned_mask)
        rec_img_3 = self.dc32(img_fea_3, u_k, learned_mask, True)

        k_to_img_3 = ifft(rec_k_3)  # convert the restored kspace to spatial domain
        img_to_k_3 = fft(rec_img_3) # convert the restored image to frequency domain


        ################################ Forth Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_4 = self.fusion31(rec_k_3, img_to_k_3)
        img_x_4 = self.fusion32(rec_img_3, k_to_img_3)

        k_fea_4, img_fea_4 = self.cnn4(*(k_x_4, img_x_4))

        rec_k_4 = self.dc41(k_fea_4, u_k, learned_mask) # rec=out2, u_k=undersample,  mask=mask, desired_sparsity=self.desired_sparsity
        rec_img_4 = self.dc42(img_fea_4, u_k, learned_mask,  True)

        k_to_img_4 = ifft(rec_k_4)  # convert the restored kspace to spatial domain
        img_to_k_4 = fft(rec_img_4) # convert the restored image to frequency domain


        ################################ Third Stage ####################################
        # fft and ifft of the restored feature to fusion

        k_x_5 = self.fusion41(rec_k_4, img_to_k_4)
        img_x_5 = self.fusion42(rec_img_4, k_to_img_4)

        k_fea_5, img_fea_5 = self.cnn5(*(k_x_5, img_x_5))

        rec_k_5 = self.dc51(k_fea_5, u_k, learned_mask)
        rec_img_5 = self.dc52(img_fea_5, u_k, learned_mask, True)

        k_to_img_5 = ifft(rec_k_5)  # convert the restored kspace to spatial domain

        out = self.fusion51(rec_img_5, k_to_img_5)

        return out
