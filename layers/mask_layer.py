"""
    Costum module for mask learning.
    By wzw.
"""
import torch
import torch as T
from torch import nn
from torch.nn.init import xavier_uniform_
import numpy as np
from torch.autograd import Function
from torch.nn import functional as F

from numpy.fft import fftshift
import scipy.io as sio
import glob
import cv2

import random

global pmask_slope, sample_slope
pmask_slope=5
sample_slope=10

class MyLayer(nn.Module):
    def __init__(self, input_size, output_size, eps=1e-12, bias=True):

        super(MyLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, 1, input_size, output_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_size))
        else:
            self.register_parameter('bias', None)
        self.activation = nn.Tanh()  
        self.variance_epsilon = eps
        self.reset_parameters()

    def reset_parameters(self):

        for p in self.parameters():  
            if p.dim() > 1:
                xavier_uniform_(p)
            else:
                nn.init.constant_(p, 0.1)

    def forward(self, x):
        u = torch.mm(x, self.weight)
        if self.bias is not None:
            u = self.bias + u
        
        return u

class ProbMask(nn.Module):
    def __init__(self, in_features=256, out_features=256, slope=pmask_slope, **kwargs):
        super(ProbMask, self).__init__(**kwargs)

        self.slope = slope
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.DoubleTensor(1, 1, in_features, out_features)) 
        self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        eps = 0.0001
        self.weight.data.uniform_(eps, 1.0-eps)
        self.weight.data = -T.log(1. / self.weight.data - 1.) / self.slope

    def forward(self, x):
        logit_weights = 0 * x[..., 0:1] + self.weight
        logit_weights = torch.sigmoid(self.slope * logit_weights)
        return logit_weights

class RescaleProbMap(nn.Module):
    def __init__(self, sparsity=1, **kwargs):
        super(RescaleProbMap, self).__init__(**kwargs)
        self.sparsity = sparsity

    def forward(self, x):
        xbar = T.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)

        
        le = (r < 1).float() 
        return le * x * r + (1-le) * (1 - (1 - x) * beta)

class RandomMask(nn.Module):
    def __init__(self, **kwargs):
        super(RandomMask, self).__init__(**kwargs)

    def forward(self, x):
        
        
        threshs = torch.rand_like(x)
        return (0*x) + threshs

class ThresholdRandomMask(nn.Module):
    def __init__(self, slope=10, **kwargs):
        super(ThresholdRandomMask, self).__init__(**kwargs)
        self.slope = None
        if slope is not None:
            self.slope=slope
        
    def forward(self, prob_mask2, thresh):
        if self.slope is not None:
            return torch.sigmoid(self.slope * (prob_mask2 - thresh))
        else:
            return prob_mask2 > thresh

class BatchThresholdRandomMaskSigmoidV1(Function):
    def __init__(self, maskType=None):
        super(BatchThresholdRandomMaskSigmoidV1, self).__init__()

    @staticmethod
    def forward(ctx, input, desired_sparsity):
    
        batch_size = len(input)
        probs = [] 
        results = [] 

        for i in range(batch_size):
            x = input[i:i+1]

            count = 0 
            while True:
                torch.manual_seed(20221105)
                prob = x.new(x.size()).uniform_()
                result = (x > prob).float()  
                if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):  
                    break

                count += 1 
                if count > 10:
                        break
            probs.append(prob)
            results.append(result)

        results = torch.cat(results, dim=0)
        probs = torch.cat(probs, dim=0)
        ctx.save_for_backward(input, probs)

        return results  

    @staticmethod
    def backward(ctx, grad_output):
        slope = 10
        input, prob = ctx.saved_tensors

        
        current_grad = slope * torch.exp(-slope * (input - prob)) / torch.pow((torch.exp(-slope*(input-prob))+1), 2)

        return current_grad * grad_output



def top_tensor(tensor, desired_sparsity):
    sample_rate = desired_sparsity
    tensor_dulp = tensor.reshape(tensor.shape[-1] * tensor.shape[-2])
    tensor_sort, index = tensor_dulp.sort()
    k = int(tensor.shape[-1] * tensor.shape[-2] * sample_rate) + 1
    the_number = tensor_sort[-k]  
    return the_number


class ThresholdRandomMaskSigmoidV1(Function):
    def __init__(self):
        super(ThresholdRandomMaskSigmoidV1, self).__init__()
    @staticmethod
    def forward(ctx, input):
        x=input
        count = 0 
        while True:
            prob = x.new(x.size()).uniform_()
            result = (x > prob).float()  

            if torch.isclose(torch.mean(result), torch.mean(x), atol=1e-3):  
                break

            count += 1 

            if count > 1000:
                
                
                break
        ctx.save_for_backward(input, prob)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        slope = 10
        input, prob = ctx.saved_tensors
        
        current_grad = slope * torch.exp(-slope * (input - prob)) / torch.pow((torch.exp(-slope*(input-prob))+1), 2)
        return current_grad * grad_output


def BatchRescaleProbMap(batch_x, sparsity):
    batch_size = len(batch_x)
    ret = []
    for i in range(batch_size):
        x = batch_x[i:i+1]
        xbar = torch.mean(x)
        r = sparsity / (xbar)
        beta = (1-sparsity) / (1-xbar)

        
        le = torch.le(r, 1).float()
        ret.append(le * x * r + (1-le) * (1 - (1 - x) * beta))

    return torch.cat(ret, dim=0)

def RescaleProbMap4SeqMRI(x, sparsity):
    xbar = T.mean(x)
    r = sparsity / xbar
    beta = (1 - sparsity) / (1 - xbar)

    
    le = (r < 1).float() 
    return le * x * r + (1-le) * (1 - (1 - x) * beta)


class ProbMask_Cartesian(nn.Module):
    def __init__(self, in_features=256, out_features=256, slope=pmask_slope, **kwargs):
        super(ProbMask_Cartesian, self).__init__(**kwargs)
        
        self.slope = slope
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.DoubleTensor(1, 1, in_features, 1))  
        self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        eps = 0.0001
        self.weight.data.uniform_(eps, 1.0-eps)
        self.weight.data = -T.log(1. / self.weight.data - 1.) / self.slope
    def forward(self, x):
        mask = torch.repeat_interleave(self.weight, repeats=x.shape[-1], dim=3)
        logit_weights = 0 * x[..., 0:1] + mask
        logit_weights = torch.sigmoid(self.slope * logit_weights)
        return logit_weights

class RescaleProbMap_Cartesian(nn.Module):
    def __init__(self, sparsity=1, **kwargs):
        super(RescaleProbMap_Cartesian, self).__init__(**kwargs)
        self.sparsity = sparsity

    def forward(self, x):
        xbar = T.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)

        
        le = (r < 1).float() 
        return le * x * r + (1-le) * (1 - (1 - x) * beta)

class RandomMask_Cartesian(nn.Module):
    def __init__(self, **kwargs):
        super(RandomMask_Cartesian, self).__init__(**kwargs)

    def forward(self, x):
        threshs = torch.rand_like(x)
        return (0*x) + threshs

class ThresholdRandomMask_Cartesian(nn.Module):
    def __init__(self, slope=10, **kwargs):
        super(ThresholdRandomMask_Cartesian, self).__init__(**kwargs)
        self.slope = None
        if slope is not None:
            self.slope=slope
        
    def forward(self, prob_mask2, thresh):
        if self.slope is not None:
            return torch.sigmoid(self.slope * (prob_mask2 - thresh))
        else:
            return prob_mask2 > thresh


class FFT(nn.Module):
    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)
    
    def forward(self, x):
        """ 
        x   - [12, 1, 240, 240]
        fft - [12, 1, 240, 240, 2]  (squeeze)-> [12, 240, 240, 2]
        """
        if np.size(x.shape) == 4:  # [12, 1, 240, 240]
            fft = torch.rfft(x, 2, onesided=False)
        elif np.size(x.shape) == 5:  # [12, 1, 240, 240,2]
            fft = torch.fft(x, 2)
        fft = fft.squeeze(1)
        fft = fft.permute(0, 3, 1, 2)
        return fft

class iFFT(nn.Module):
    def __init__(self, **kwargs):
        super(iFFT, self).__init__(**kwargs)
    
    def forward(self, x):
        xt = x.permute(0, 2, 3, 1)
        ifft = torch.ifft(xt, 2)
        ifft = ifft.permute(0, 3, 1, 2)
        return ifft

class ConcatenateZero(nn.Module):

    def __init__(self, **kwargs):
        super(ConcatenateZero, self).__init__(**kwargs)
    
    def forward(self, x):
        pass

class UnderSample(nn.Module):
    def __init__(self, **kwargs):
        super(UnderSample, self).__init__(**kwargs)

    def forward(self, fft, mask):
        ufft = torch.zeros_like(fft)
        ufft[:, 0:1, ...] = fft[:, 0:1, ...] * mask
        ufft[:, 1:2, ...] = fft[:, 1:2, ...] * mask
        return ufft

class ComplexAbs(nn.Module):
    def __init__(self, **kwargs):
        super(ComplexAbs, self).__init__(**kwargs)

    def forward(self, x):
        """ pytorch 1.5.0 Add complex32, complex64 and complex128 dtypes,
        but got no attribute 'Complex64'
        """
        x_abs = torch.sqrt(x[:, 0:1, ...]**2 + x[:, 1:2, ...]**2)
        return x_abs

class PixelTo255(nn.Module):
    def __init__(self, **kwargs):
        super(PixelTo255, self).__init__(**kwargs)

    def forward(self, x):
        out = torch.zeros_like(x)
        for i in range(x.shape[1]):
            out[:, i:i+1, ...] = 255 * (x[:, i:i+1, ...] - torch.min(x[:, i:i+1, ...])) / (torch.max(x[:, i:i+1, ...]) - torch.min(x[:, i:i+1, ...]))
        return out

class Mask_Layer(nn.Module):
    def __init__(self, inputs_size=[256, 256], pmask_slope=5, desired_sparsity=0.125, sample_slope=10):
        super(Mask_Layer, self).__init__()
        self.layer_probmask = ProbMask(inputs_size[0], inputs_size[1], slope=pmask_slope)  
        self.layer_rescale = RescaleProbMap(sparsity=desired_sparsity)  
        self.layer_randommask = RandomMask() 
        self.layer_thresh = ThresholdRandomMask(slope=sample_slope)  
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()

        print("random_model, pmask_slope={}, desired_sparsity={}, sample_slope={}".format(pmask_slope, desired_sparsity, sample_slope))

    def forward(self, x):    
        prob_mask1 = self.layer_probmask(x)  
        prob_mask2 = self.layer_rescale(prob_mask1)  
        thresh = self.layer_randommask(prob_mask2)  
        mask = self.layer_thresh(prob_mask2, thresh)  
        fft = self.layer_fft(x)   
        u_k = self.layer_undersample(fft, mask)  
        
        
        uifft = self.layer_ifft(u_k)  
        complex_abs = self.layer_abs(uifft)  
        
        return (uifft, complex_abs, mask, fft, u_k)
   
class Mask_Fixed_Layer(nn.Module):
    def __init__(self, ckpt, desired_sparsity, inputs_size=[256, 256], pmask_slope=5, sample_slope=10):
        super(Mask_Fixed_Layer, self).__init__()
        self.desired_sparsity = desired_sparsity
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        
        self.ckpt = ckpt
        self.layer_probmask = ProbMask(inputs_size[0], inputs_size[1], slope=pmask_slope)  
        
        self.layer_rescale = RescaleProbMap(sparsity=desired_sparsity)  
        self.layer_randommask = RandomMask() 
        self.layer_thresh = ThresholdRandomMask(slope=sample_slope)  
        self.weights = self.get_weights()
        self.zero_one = self.get_zero_one()
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        print("Mask_Fixed_Layer, pmask_slope={}, desired_sparsity={}, sample_slope={}".format(pmask_slope, desired_sparsity, sample_slope))
        print("ckpt={} in Mask learning loading".format(self.ckpt))

    def forward(self, x):
        if isinstance(self.zero_one, np.ndarray):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mask = torch.from_numpy(self.zero_one).to(device)
        fft = self.layer_fft(x)   
        undersample = self.layer_undersample(fft, mask)  
        uifft = self.layer_ifft(undersample)  
        complex_abs = self.layer_abs(uifft)  
        return (uifft, complex_abs, mask, fft, undersample)

    def top_tensor(self, tensor):
        sample_rate = self.desired_sparsity
        tensor_dulp = tensor.reshape(tensor.shape[-1] * tensor.shape[-2])
        tensor_sort, index = tensor_dulp.sort()
        k = int(tensor.shape[-1] * tensor.shape[-2] * sample_rate)
        the_number = tensor_sort[-k]  
        return the_number
    
    def get_parameter(self, tensor):
        
        mask = tensor[0:1, ...]
        the_number = self.top_tensor(mask)
        
        mask[mask >= the_number] = 1
        mask[mask < the_number] = 0
        
        return mask

    def get_weights(self):
        pre_trained_model = torch.load(self.ckpt)
        for key, value in pre_trained_model.items():
            break
        return value

    def probmask(self, weight, x):
        slope = self.pmask_slope
        logit_weights = 0 * x[..., 0:1] + weight
        logit_weights = torch.sigmoid(slope * logit_weights)
        return logit_weights

    def get_zero_one(self):
        x = torch.rand_like(self.weights)
        prob_mask = self.probmask(self.weights, x)
        zero_one = self.get_parameter(prob_mask)
        return zero_one.detach().cpu().numpy()

class Mask_CartesianLayer(nn.Module):
    def __init__(self, inputs_size=[256, 256], pmask_slope=5, desired_sparsity=0.125, sample_slope=10):
        super(Mask_CartesianLayer, self).__init__()
        self.layer_probmask = ProbMask_Cartesian(inputs_size[0], inputs_size[1], slope=pmask_slope)  
        self.layer_rescale = RescaleProbMap_Cartesian(sparsity=desired_sparsity)  
        self.layer_randommask = RandomMask_Cartesian() 
        self.layer_thresh = ThresholdRandomMask_Cartesian(slope=sample_slope)  
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        print("pmask_slope={}, desired_sparsity={}, sample_slope={}".format(pmask_slope, desired_sparsity, sample_slope))
        print("Cartesian learning")

    def forward(self, x):    
        prob_mask1 = self.layer_probmask(x)  
        prob_mask2 = self.layer_rescale(prob_mask1)  
        thresh = self.layer_randommask(prob_mask2)  
        mask = self.layer_thresh(prob_mask2, thresh)  
        fft = self.layer_fft(x)   
        undersample = self.layer_undersample(fft, mask)  
        
        
        uifft = self.layer_ifft(undersample)  
        complex_abs = self.layer_abs(uifft)  
        
        return (uifft, complex_abs, mask, fft, undersample)

class Mask_Fixed_CartesianLayer(nn.Module):
    def __init__(self, ckpt, desired_sparsity, inputs_size=[256, 256], pmask_slope=5, sample_slope=10):
        super(Mask_Fixed_CartesianLayer, self).__init__()
        self.desired_sparsity = desired_sparsity
        self.pmask_slope = pmask_slope
        self.sample_slope = sample_slope
        
        self.ckpt = ckpt
        self.layer_probmask = ProbMask_Cartesian(inputs_size[0], inputs_size[1], slope=pmask_slope)  
        
        self.layer_rescale = RescaleProbMap_Cartesian(sparsity=desired_sparsity)  
        self.layer_randommask = RandomMask_Cartesian() 
        self.layer_thresh = ThresholdRandomMask_Cartesian(slope=sample_slope)  
        self.weights = self.get_weights()
        self.zero_one = self.get_zero_one()
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        print("Mask_Fixed_Layer, pmask_slope={}, desired_sparsity={}, sample_slope={}".format(pmask_slope, desired_sparsity, sample_slope))
        print("ckpt={} in Mask learning loading".format(self.ckpt))

    def forward(self, x):  
        if isinstance(self.zero_one, np.ndarray):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mask = torch.from_numpy(self.zero_one).to(device)
        fft = self.layer_fft(x)   
        undersample = self.layer_undersample(fft, mask)  
        uifft = self.layer_ifft(undersample)  
        complex_abs = self.layer_abs(uifft)  
        return (uifft, complex_abs, mask, fft, undersample)

    def top_tensor(self, tensor):
        sample_rate = self.desired_sparsity
        tensor_dulp = tensor.reshape(tensor.shape[-1] * tensor.shape[-2])
        tensor_sort, index = tensor_dulp.sort()
        k = int(tensor.shape[-1] * tensor.shape[-2] * sample_rate) + 1
        the_number = tensor_sort[-k]  
        return the_number
    
    def get_parameter(self, tensor):
        
        mask = tensor[0:1, ...]
        the_number = self.top_tensor(mask)
        
        mask[mask >= the_number] = 1
        mask[mask < the_number] = 0
        
        mask_2d = torch.repeat_interleave(mask, repeats=mask.shape[2], dim=3)
        return mask_2d

    def get_weights(self):
        pre_trained_model = torch.load(self.ckpt)
        for key, value in pre_trained_model.items():
            break
        return value

    def probmask(self, weight, x):
        slope = self.pmask_slope
        logit_weights = 0 * x[..., 0:1] + weight
        logit_weights = torch.sigmoid(slope * logit_weights)
        return logit_weights

    def get_zero_one(self):
        x = torch.rand_like(self.weights)
        prob_mask = self.probmask(self.weights, x)
        zero_one = self.get_parameter(prob_mask)
        return zero_one.detach().cpu().numpy()


class Mask_Linear(nn.Module):
    def __init__(self, inputs_size=[256, 256]):
        super(Mask_Linear, self).__init__()
        self.mask = torch.zeros(inputs_size[0]).uniform_(0.0, 1.0)
        self._initmask()
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        

    def _initmask(self):
        self.mask[self.mask > 0.8] = True
        self.mask[self.mask <= 0.8] = False
        self.mask = self.mask.repeat(256, 1)
        self.mask = self.mask.unsqueeze(0)
        self.mask = self.mask.unsqueeze(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask = self.mask.to(device)

    def forward(self, x):
        mask = self.mask
        fft = self.layer_fft(x)   
        undersample = self.layer_undersample(fft, mask)  
        uifft = self.layer_ifft(undersample)  
        complex_abs = self.layer_abs(uifft)  
        return (uifft, complex_abs, mask)

class Mask_Oneshot_Layer(nn.Module):
    def __init__(self, inputs_size=[256, 256], pmask_slope=5, desired_sparsity=0.125, sample_slope=10):
        super(Mask_Oneshot_Layer, self).__init__()
        self.layer_probmask = ProbMask(inputs_size[0], inputs_size[1], slope=pmask_slope)  
        self.layer_rescale = RescaleProbMap(sparsity=desired_sparsity)  
        
        self.layer_thresh = ThresholdRandomMaskSigmoidV1.apply  
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        print("random_model, pmask_slope={}, desired_sparsity={}, sample_slope={}".format(pmask_slope, desired_sparsity, sample_slope))

    def forward(self, x):    
        prob_mask1 = self.layer_probmask(x)  
        prob_mask2 = self.layer_rescale(prob_mask1)  
        
        mask = self.layer_thresh(prob_mask2)  
        fft = self.layer_fft(x)   
        u_k = self.layer_undersample(fft, mask)  
        
        
        uifft = self.layer_ifft(u_k)  
        complex_abs = self.layer_abs(uifft)  
        
        return (uifft, complex_abs, mask, fft, u_k)

class Mask_Oneshot_1D_Layer(nn.Module):
    def __init__(self, inputs_size=[256, 256], pmask_slope=5, desired_sparsity=0.125, sample_slope=10):
        super(Mask_Oneshot_1D_Layer, self).__init__()
        self.layer_probmask = ProbMask(inputs_size[0], inputs_size[1], slope=pmask_slope)  
        
        self.layer_rescale = RescaleProbMap(sparsity=desired_sparsity)  
        
        self.layer_randommask = RandomMask() 
        self.layer_thresh = ThresholdRandomMaskSigmoidV1.apply  
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        self.H, self.W = inputs_size

        print("random_model, pmask_slope={}, desired_sparsity={}, sample_slope={}".format(pmask_slope, desired_sparsity, sample_slope))

    def forward(self, x):    
        N = x.shape[0]
        prob_mask1 = self.layer_probmask(x)  

        prob_mask1 = self.change_mask2dto1d(prob_mask1)

        prob_mask2 = self.layer_rescale(prob_mask1)  
        
        mask = self.layer_thresh(prob_mask2)  
        
        fft = self.layer_fft(x)   
        u_k = self.layer_undersample(fft, mask)  
        
        
        uifft = self.layer_ifft(u_k)  
        complex_abs = self.layer_abs(uifft)  
        
        return (uifft, complex_abs, mask, fft, u_k)

    def change_mask2dto1d(self, out):
        out = torch.sum(out, dim=2, keepdims=True)
        out = F.softplus(out) 
        return out

class Hand_Tailed_Mask_Layer(nn.Module):
    def __init__(self, desired_sparsity, traj_type, inputs_size, resize_size=None, for_bsa=False):
        super(Hand_Tailed_Mask_Layer, self).__init__()
        desired_sparsity = str(int(desired_sparsity*100))
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        self.epoch_idx=None

        self.trajectories_b_c_h_w = self._load_mask(desired_sparsity, traj_type, inputs_size, resize_size=resize_size, for_bsa=for_bsa)
        print(" desired_sparsity={}".format(desired_sparsity))

    @staticmethod
    def _load_mask(desired_sparsity, traj_type, inputs_size, resize_size, for_bsa):
        """
        desired_sparsity should be 0 ~ 1
        """
        acc = int(desired_sparsity)
        data_type = 'brain'
        mask_type = traj_type
        
        img_size = inputs_size
        
        if for_bsa:
            # file_path_pattern1 = "data/masks/motion33mask/bsa/gaussian/cartesian_ablation2_240_240_*.mat"
            file_path_pattern1 = "data/masks/motion33mask/bsa/random/cartesian_240_240_33*.mat"
            # all_masks_path = 20*file_paths1[:2]
            # file_path_pattern1 = "data/masks/motion33mask/ablation2/gaussian/cartesian_ablation2_240_240_*.mat"
        else:
            file_path_pattern1 = "./data/masks/motion{}mask/{}_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        file_paths1 = glob.glob(file_path_pattern1)

        # file_path_pattern2 = "./data/masks/motion{}mask/{}_smallcenter_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        # file_paths2 = glob.glob(file_path_pattern2)
        # all_masks_path = file_paths1 + file_paths2
        all_masks_path = file_paths1
        
        if for_bsa:
            all_masks_path = 20*file_paths1[:2]
            
        if resize_size:
            img_size = resize_size
        trajectories_b_c_h_w = np.zeros((len(all_masks_path), 1, img_size[0], img_size[0]))
        i = 0
        for path in all_masks_path:
            trajectory = sio.loadmat(path)['Umask']
            trajectory = fftshift(trajectory, axes=(-2, -1))
            if resize_size[0] != 240:
                trajectory = cv2.resize(trajectory, resize_size)

            trajectories_b_c_h_w[i] = trajectory
            i = i + 1      
        

        trajectories_b_c_h_w = torch.tensor(trajectories_b_c_h_w)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trajectories_b_c_h_w = trajectories_b_c_h_w.to(device, dtype=torch.float)

        print("****************sparity = {}****************".format(np.mean(trajectory)))
        return trajectories_b_c_h_w

    def forward(self, x, *args):
        fft = self.layer_fft(x)
        if self.epoch_idx is not None:
            idx = self.epoch_idx
        else:
            idx = torch.randperm(self.trajectories_b_c_h_w.shape[0])[:x.shape[0]]
        batch_trajectories = self.trajectories_b_c_h_w[idx]

        # add 2d mask.
        # mask_2d = (batch_trajectories.new(batch_trajectories.size()).uniform_()>0.2).float()
        # batch_trajectories = batch_trajectories* mask_2d

        undersample = self.layer_undersample(fft, batch_trajectories)  
        
        uifft = self.layer_ifft(undersample)  

        # for ind_batch in uifft.shape[0]:

        complex_abs = self.layer_abs(uifft)  
        output = {'masked_k': undersample, 'complex_abs': complex_abs, 'zero_filling': uifft, 'batch_trajectories': batch_trajectories}

        return output

class Hand_Tailed_Mask_Singleshot_Layer(nn.Module):
    def __init__(self, desired_sparsity, traj_type, inputs_size, resize_size=None, for_bsa=False):
        super(Hand_Tailed_Mask_Singleshot_Layer, self).__init__()
        desired_sparsity = str(int(desired_sparsity*100))
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        self.epoch_idx=None

        self.trajectories_b_c_h_w = self._load_mask(desired_sparsity, traj_type, inputs_size, resize_size=resize_size, for_bsa=for_bsa)
        print(" desired_sparsity={}".format(desired_sparsity))

    @staticmethod
    def _load_mask(desired_sparsity, traj_type, inputs_size, resize_size, for_bsa):
        """
        desired_sparsity should be 0 ~ 1
        """
        acc = int(desired_sparsity)
        data_type = 'brain'
        mask_type = traj_type
        
        img_size = inputs_size
        
        
        file_path_pattern1 = "./data/masks/motion{}mask/{}_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        file_paths1 = glob.glob(file_path_pattern1)

        # file_path_pattern2 = "./data/masks/motion{}mask/{}_smallcenter_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        # file_paths2 = glob.glob(file_path_pattern2)
        # all_masks_path = file_paths1 + file_paths2
        all_masks_path = file_paths1
        
        if for_bsa:
            all_masks_path = 20*file_paths1[:2]
            
        if resize_size:
            img_size = resize_size
        trajectories_b_c_h_w = np.zeros((len(all_masks_path), 1, img_size[0], img_size[0]))
        i = 0
        for path in all_masks_path:
            trajectory = sio.loadmat(path)['Umask']
            trajectory = fftshift(trajectory, axes=(-2, -1))
            if resize_size[0] != 240:
                trajectory = cv2.resize(trajectory, resize_size)

            trajectories_b_c_h_w[i] = trajectory
            i = i + 1      
        

        trajectories_b_c_h_w = torch.tensor(trajectories_b_c_h_w)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trajectories_b_c_h_w = trajectories_b_c_h_w.to(device, dtype=torch.float)

        print("****************sparity = {}****************".format(np.mean(trajectory)))
        return trajectories_b_c_h_w

    def forward(self, x, *args):
        fft = self.layer_fft(x)
        if self.epoch_idx is not None:
            idx = self.epoch_idx
        else:
            idx = torch.randperm(self.trajectories_b_c_h_w.shape[0])[:x.shape[0]]
        batch_trajectories = self.trajectories_b_c_h_w[idx] # [B,C,H,W]

        # add 2d mask.
        # mask_2d = (batch_trajectories.new(batch_trajectories.size()).uniform_()>0.2).float()
        # batch_trajectories = batch_trajectories* mask_2d


        # Define the shape of the masks
        B, C, H = batch_trajectories.shape[0], batch_trajectories.shape[1], batch_trajectories.shape[2]
        dropout_ratio = 0.02
        # Generate the mask1
        mask1 = torch.ones((B, C, H))
        # Set 1 in 10% of the positions along the last dimension for each batch
        num_elements_per_batch = C * H
        num_ones_per_batch = int(num_elements_per_batch * dropout_ratio)

        for b in range(B):
            indices = torch.randperm(num_elements_per_batch)[:num_ones_per_batch]
            mask1[b].view(-1)[indices] = 0
        # Expand mask1 to match the shape of mask2
        mask2 = mask1.unsqueeze(-1).expand(B, C, H, H).cuda()

        batch_trajectories = mask2 * batch_trajectories # drop_out for single-shot

        undersample = self.layer_undersample(fft, batch_trajectories)  
        uifft = self.layer_ifft(undersample)  
        # for ind_batch in uifft.shape[0]:
        complex_abs = self.layer_abs(uifft)  
        output = {'masked_k': undersample, 'complex_abs': complex_abs, 'zero_filling': uifft, 'batch_trajectories': batch_trajectories}
        return output

class Hand_Tailed_Mask_Layer_maskablation(nn.Module):
    def __init__(self, ablation_which, ablation_var, resize_size=[240,240]):
        super(Hand_Tailed_Mask_Layer_maskablation, self).__init__()
        
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()
        self.epoch_idx=None

        self.trajectories_b_c_h_w = self._load_mask(ablation_which, ablation_var, resize_size=resize_size)
        

    @staticmethod
    def _load_mask(ablation_which, ablation_var, resize_size):
        """
        desired_sparsity should be 0 ~ 1
        """
        
        file_path_pattern1 = "./data/masks/motion33mask/{}/{}/*.mat".format(ablation_which, ablation_var)
        file_paths1 = glob.glob(file_path_pattern1)

        # file_path_pattern2 = "./data/masks/motion{}mask/{}_smallcenter_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        # file_paths2 = glob.glob(file_path_pattern2)
        # all_masks_path = file_paths1 + file_paths2
        all_masks_path = file_paths1

            
        if resize_size:
            img_size = resize_size
        trajectories_b_c_h_w = np.zeros((len(all_masks_path), 1, img_size[0], img_size[0]))
        i = 0
        for path in all_masks_path:
            trajectory = sio.loadmat(path)['Umask']
            trajectory = fftshift(trajectory, axes=(-2, -1))
            if resize_size[0] != 240:
                trajectory = cv2.resize(trajectory, resize_size)

            trajectories_b_c_h_w[i] = trajectory
            i = i + 1      
        

        trajectories_b_c_h_w = torch.tensor(trajectories_b_c_h_w)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trajectories_b_c_h_w = trajectories_b_c_h_w.to(device, dtype=torch.float)

        print("****************sparity = {}****************".format(np.mean(trajectory)))
        return trajectories_b_c_h_w

    def forward(self, x, *args):
        fft = self.layer_fft(x)
        if self.epoch_idx is not None:
            idx = self.epoch_idx
        else:
            idx = torch.randperm(self.trajectories_b_c_h_w.shape[0])[:x.shape[0]]
        batch_trajectories = self.trajectories_b_c_h_w[idx]

        # add 2d mask.
        # mask_2d = (batch_trajectories.new(batch_trajectories.size()).uniform_()>0.2).float()
        # batch_trajectories = batch_trajectories* mask_2d

        undersample = self.layer_undersample(fft, batch_trajectories)  
        
        uifft = self.layer_ifft(undersample)  

        # for ind_batch in uifft.shape[0]:

        complex_abs = self.layer_abs(uifft)  
        output = {'masked_k': undersample, 'complex_abs': complex_abs, 'zero_filling': uifft, 'batch_trajectories': batch_trajectories}

        return output


class Hand_Tailed_Mask_LayerAP_RL(nn.Module):
    # Can 50% choose AP or RL.
    def __init__(self, desired_sparsity, traj_type, inputs_size, resize_size=None, for_bsa=False):
        super(Hand_Tailed_Mask_LayerAP_RL, self).__init__()
        desired_sparsity = str(int(desired_sparsity*100))
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()

        self.trajectories_b_c_h_w = self._load_mask(desired_sparsity, traj_type, inputs_size, resize_size=resize_size, for_bsa=for_bsa)
        print(" desired_sparsity={}".format(desired_sparsity))

    @staticmethod
    def _load_mask(desired_sparsity, traj_type, inputs_size, resize_size, for_bsa):
        """
        desired_sparsity should be 0 ~ 1
        """
        acc = int(desired_sparsity)
        data_type = 'brain'
        mask_type = traj_type
        
        img_size = inputs_size
        
        
        all_masks_path = "./data/masks/motion{}mask/{}_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        all_masks_path = glob.glob(all_masks_path)
        if for_bsa:
            all_masks_path = 20*all_masks_path[:2]
            
        if resize_size:
            img_size = resize_size
        trajectories_b_c_h_w = np.zeros((len(all_masks_path), 1, img_size[0], img_size[0]))
        i = 0
        for path in all_masks_path:
            trajectory = sio.loadmat(path)['Umask']
            trajectory = fftshift(trajectory, axes=(-2, -1))
            if resize_size[0] != 240:
                trajectory = cv2.resize(trajectory, resize_size)

            trajectories_b_c_h_w[i] = trajectory
            i = i + 1      
        

        trajectories_b_c_h_w = torch.tensor(trajectories_b_c_h_w)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trajectories_b_c_h_w = trajectories_b_c_h_w.to(device, dtype=torch.float)

        print("****************sparity = {}****************".format(np.mean(trajectory)))
        return trajectories_b_c_h_w

    def forward(self, x, *args, **kwargs):
        fft = self.layer_fft(x)
        idx = torch.randperm(self.trajectories_b_c_h_w.shape[0])[:x.shape[0]]
        batch_trajectories = self.trajectories_b_c_h_w[idx]

        # if random.random > 0.6: # change x,y -> y,x for change AP -> RL 
        # if torch.rand(1) > 0.6: # change x,y -> y,x for change AP -> RL 
        if kwargs['RL_seed'] > 0.6:
            batch_trajectories = batch_trajectories.permute(0,1,3,2)
        # add 2d mask.
        # mask_2d = (batch_trajectories.new(batch_trajectories.size()).uniform_()>0.2).float()
        # batch_trajectories = batch_trajectories* mask_2d

        undersample = self.layer_undersample(fft, batch_trajectories)  
        
        uifft = self.layer_ifft(undersample)  

        # for ind_batch in uifft.shape[0]:

        complex_abs = self.layer_abs(uifft)  
        output = {'masked_k': undersample, 'complex_abs': complex_abs, 'zero_filling': uifft, 'batch_trajectories': batch_trajectories}

        return output


class Hand_Tailed_Mask_Layer4SSL(nn.Module):
    def __init__(self, desired_sparsity, traj_type, inputs_size, resize_size=None):
        super(Hand_Tailed_Mask_Layer4SSL, self).__init__()
        desired_sparsity = str(int(desired_sparsity*100))
        self.layer_fft = FFT()
        self.layer_ifft = iFFT()
        self.layer_undersample = UnderSample()
        self.layer_abs = ComplexAbs()

        self.trajectories_b_c_h_w = self._load_mask(desired_sparsity, traj_type, inputs_size, resize_size=resize_size)
        print(" desired_sparsity={}".format(desired_sparsity))

    @staticmethod
    def _load_mask(desired_sparsity, traj_type, inputs_size, resize_size):
        """
        desired_sparsity should be 0 ~ 1
        """
        acc = int(desired_sparsity)
        data_type = 'brain'
        mask_type = traj_type
        
        img_size = inputs_size
        
        
        all_masks_path = "./data/masks/motion{}mask/{}_{}_{}_{}_*.mat".format(acc, mask_type, img_size[0], img_size[1], acc)
        all_masks_path = glob.glob(all_masks_path)
        if resize_size:
            img_size = resize_size
        trajectories_b_c_h_w = np.zeros((len(all_masks_path), 1, img_size[0], img_size[0]))
        i = 0
        for path in all_masks_path:
            trajectory = sio.loadmat(path)['Umask']
            trajectory = fftshift(trajectory, axes=(-2, -1))
            if resize_size[0] != 240:
                trajectory = cv2.resize(trajectory, resize_size)

            trajectories_b_c_h_w[i] = trajectory
            i = i + 1      
        

        trajectories_b_c_h_w = torch.tensor(trajectories_b_c_h_w)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trajectories_b_c_h_w = trajectories_b_c_h_w.to(device, dtype=torch.float)

        print("****************sparity = {}****************".format(np.mean(trajectory)))
        return trajectories_b_c_h_w

    def forward(self, x, mask):
        fft = self.layer_fft(x)
        idx = torch.randperm(self.trajectories_b_c_h_w.shape[0])[:x.shape[0]]
        batch_trajectories = self.trajectories_b_c_h_w[idx]

        batch_trajectories = batch_trajectories * mask
        # add 2d mask.
        # mask_2d = (batch_trajectories.new(batch_trajectories.size()).uniform_()>0.2).float()
        # batch_trajectories = batch_trajectories* mask_2d

        undersample = self.layer_undersample(fft, batch_trajectories)  
        
        uifft = self.layer_ifft(undersample)  

        # for ind_batch in uifft.shape[0]:

        complex_abs = self.layer_abs(uifft)  
        output = {'masked_k': undersample, 'complex_abs': complex_abs, 'zero_filling': uifft, 'batch_trajectories': batch_trajectories}

        return output
    
if __name__ == "__main__":

    x = torch.zeros(8, 1, 240,240)
    x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    samplenet = Hand_Tailed_Mask_Singleshot_Layer(0.33, 'cartesian', (240, 240), resize_size=[240,240])

    out = samplenet(x)
    klkl =1