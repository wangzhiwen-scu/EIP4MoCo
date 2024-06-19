
import torch
from torch import nn
from torch.nn import functional as F
import sys
from torch.autograd import Function

class Sampler(nn.Module):
    def __init__(self, shape, line_constrained, mini_batch):
        super().__init__()
        

        self.mask_net = Sampler2D()
        self.rescale = BatchRescaleProbMap
        self.binarize = BatchThresholdRandomMaskSigmoidV1.apply
        self.shape = shape
        self.mini_batch = mini_batch
        self.line_constrained = line_constrained
    def forward(self, full_kspace, observed_kspace, old_mask, budget):
        sparsity = budget / (self.shape[0] * self.shape[1]) if not self.line_constrained else (budget / self.shape[0]) 
        temp = torch.cat([observed_kspace, full_kspace*old_mask], dim=1)

        mask = self.mask_net(temp, old_mask)
        binary_mask = self.binarize(mask, sparsity)
        binary_mask = old_mask + binary_mask
        binary_mask = torch.clamp(binary_mask, min=0, max=1)
        masked_kspace = binary_mask * full_kspace
        return masked_kspace, binary_mask

class ConvBlock(nn.Module):

    def __init__(self, in_chans, out_chans, drop_prob):

        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):

        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class Sampler2D(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1], fixed_input=False, roi_kspace=False):
        super(Sampler2D, self).__init__()
        
        in_chans = 5
        if roi_kspace:
            in_chans = 7
        out_chans = 1
        chans = 64
        num_pool_layers = 4
        drop_prob = 0

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch, drop_prob)

        self.up_sample_layers = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_sample_layers += [ConvBlock(ch * 2, ch // 2, drop_prob)]
            ch //= 2
        self.up_sample_layers += [ConvBlock(ch * 2, ch, drop_prob)]
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=1),
            nn.Conv2d(ch // 2, out_chans, kernel_size=1),
            nn.Conv2d(out_chans, out_chans, kernel_size=1),
        )

        self.fixed_input = fixed_input
        if fixed_input:
            print('generate random input tensor')
            fixed_input_tensor = torch.randn(size=[1, 5, 128, 128])
            self.fixed_input_tensor = nn.Parameter(fixed_input_tensor, requires_grad=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        
        x = torch.cat([x, mask], dim=1)

        if self.fixed_input:
            x = self.fixed_input_tensor

        output = x 
        
        stack = []
        

        
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.max_pool2d(output, kernel_size=2)

        output = self.conv(output)

        
        for layer in self.up_sample_layers:
            downsample_layer = stack.pop()
            layer_size = (downsample_layer.shape[-2], downsample_layer.shape[-1])
            output = F.interpolate(output, size=layer_size, mode='bilinear', align_corners=False)
            output = torch.cat([output, downsample_layer], dim=1)
            output = layer(output)

        out = self.conv2(output)

        out = F.softplus(out) 

        out = out / torch.max(out.reshape(out.shape[0], -1), dim=1)[0].reshape(-1, 1, 1, 1)

        
        new_mask = out * (1-mask)

        return new_mask
    
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
    
def rfft(input):
    fft = torch.rfft(input, 2, onesided=False)  
    fft = fft.squeeze(1)
    fft = fft.permute(0, 3, 1, 2)
    return fft  

def fft(input):
    
    
    input = input.permute(0, 2, 3, 1)
    input = torch.fft(input, 2, normalized=False) 

    
    input = input.permute(0, 3, 1, 2)
    return input

def ifft(input):
    input = input.permute(0, 2, 3, 1)
    input = torch.ifft(input, 2, normalized=False) 
    
    input = input.permute(0, 3, 1, 2)
    return input