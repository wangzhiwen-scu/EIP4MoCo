import torch
from kornia.filters import sobel



img = torch.randn(4,1,8,8)

edge = sobel(img)

print(edge.shape)