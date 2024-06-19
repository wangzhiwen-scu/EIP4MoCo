# torch version == 1.13.1
import torch

print(torch.__version__)

x = torch.tensor([[1,2,3,4], [5,6,7,8]] ,dtype=torch.complex64)
print(x)

fft2 = torch.fft.ifft2(x)
print(fft2)

ifft2 = torch.fft.fft2(fft2)

print(ifft2)
