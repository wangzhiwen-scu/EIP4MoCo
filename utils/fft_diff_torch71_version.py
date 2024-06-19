# torch version == 1.7.1
#!/usr/bin/env ~miniconda3/envs/torch/bin/python
# -*- coding: cp1252 -*-
import torch

def rfft(input):
    fft = torch.rfft(input, 2, onesided=False)  # Real-to-complex Discrete Fourier Transform. normalized =False
    fft = fft.squeeze(1)
    fft = fft.permute(0, 3, 1, 2)
    return fft  

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


print(torch.__version__)

# Do not use rfft when torc==1.7.1, the results maybe wrong.
x = torch.tensor([[[1,0],[2,0],[3,0],[4,0]], [[5,0],[6,0],[7,0],[8,0]]])*1.0
print(x)

fft2 = torch.fft(x,2, normalized=False) # !
print(fft2)

ifft2 = torch.ifft(fft2, 2, normalized=False) # !

print(ifft2)