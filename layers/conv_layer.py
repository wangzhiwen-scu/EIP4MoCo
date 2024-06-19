import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
        )
    
    def forward(self, input):
        return self.conv(input)

class DuDoRnetDC(nn.Module):
    def __init__(self):
        super(DuDoRnetDC, self).__init__()
        self.v = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        print('\nusing DuDoRnetDC Layer.')
    
    def forward(self, rec, u_k, mask, desired_sparsity):
        """
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

class DC(nn.Module):
    def __init__(self):
        super(DC, self).__init__()
        self.w = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
    
    def forward(self, rec, u_k, mask, is_img=False):
        if is_img:
            rec = fft(rec)
        result = mask * (rec * self.w / (1 + self.w) + u_k * 1 / (self.w + 1)) 
        result = result + (1 - mask) * rec 

        if is_img:
            result = ifft(result)
        
        return result

    def get_weights(self):
        value = None
        pre_trained_model = torch.load(self.ckpt)
        for key, value in pre_trained_model.items():
            break
        return value

def torch_binarize(k_data, desired_sparsity):
    batch, channel, x_h, x_w = k_data.shape
    tk = int(desired_sparsity*x_h*x_w)+1
    k_data = k_data.reshape(batch, channel, x_h*x_w, 1)
    values, indices = torch.topk(k_data, tk, dim=2)
    k_data_binary =  (k_data >= torch.min(values))
    k_data_binary = k_data_binary.reshape(batch, channel, x_h, x_w).float()
    return k_data_binary

def  fft(x):
    x = torch.rfft(x, 2, onesided=False).squeeze(1)  
    x = x.permute(0, 3, 1, 2)
    
    return x

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2)  
    
    input = input.permute(0, 3, 1, 2) 
    return input