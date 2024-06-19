import sys
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from torchvision import datasets, transforms


sys.path.append('.')

from layers.drt_layers import TransformerBlock, PatchEmbed, PatchUnEmbed

#patch embedding -> transformer -> patch unembedding
class BasicBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, patch_size):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        H, W = self.input_resolution
        self.num_heads = num_heads
        self.transformer = TransformerBlock(self.dim, (H//self.patch_size, W//self.patch_size),
                                            num_heads) #patch size is 4
    
    def forward(self, x):
        x=self.transformer(x)
        return x

#Basic Block -> basic block (with skip connection)
class ResidualLayer(nn.Module):
    def __init__(self, dim, input_resolution, residual_depth, patch_size):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.residual_depth = residual_depth
        self.input_resolution = input_resolution
        self.block1 = BasicBlock(self.dim, self.input_resolution, 2, self.patch_size) #multi-heads: 2
        self.block2 = BasicBlock(self.dim, self.input_resolution, 2, self.patch_size)
        self.conv_out = nn.Conv2d(self.dim, self.dim, 3, padding = 1)
        
    def forward(self, x):
        B, HW, C = x.shape
        H, W = self.input_resolution
        shortcut = x
        for _ in range(self.residual_depth):
            x = self.block1(self.block2(x))
            x = torch.add(x, shortcut)
        #convolution at the end of each residual block
        x = x.transpose(1,2).view(B, C, H//self.patch_size, W//self.patch_size)
        x = self.conv_out(x).flatten(2).transpose(1,2)#B L C
        return x
    
#recursive network based on residual units
class DeepRecursiveTransformer(nn.Module):
    def __init__(self, dim, input_resolution, patch_size, residual_depth, recursive_depth, in_channels):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.recursive_depth = recursive_depth
        self.input_resolution = input_resolution
        self.residual_depth = residual_depth
        self.H, self.W = self.input_resolution
        assert self.H == self.W, "Input hight and width should be the same"
        self.input_conv1 = nn.Conv2d(in_channels, self.dim, 3, padding=1)
        self.patch_embed = PatchEmbed(img_size=self.H, patch_size = self.patch_size,
                                      in_chans=in_channels, embed_dim=self.dim)
        self.patch_unembed = PatchUnEmbed(img_size=self.H, patch_size = self.patch_size,
                                          in_chans=self.dim, unembed_dim=in_channels)
        self.recursive_layers = nn.ModuleList()
        for i in range(self.recursive_depth):
            layer = ResidualLayer(self.dim, self.input_resolution, self.residual_depth, self.patch_size)
            self.recursive_layers.append(layer)
        self.output_conv1 = nn.Conv2d(self.dim, in_channels, 3, padding=1)
        #use imagenet mean and std for general domain normalisation
        # self.normalise_layer = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # self.denormalise_layer = transforms.Normalize((-0.485, -0.456, -0.406), (1./0.229, 1./0.224, 1./0.225))

        self.normalise_layer = transforms.Normalize((0.485), (0.229))
        self.denormalise_layer = transforms.Normalize((-0.485), (1./0.229))
        self.apply(self._init_weights)
        self.activation = nn.LeakyReLU()
        
    #weight initialisation scheme
    def _init_weights(self, l):
        if isinstance(l, nn.Linear):
            trunc_normal_(l.weight, std=.02)
            if isinstance(l, nn.Linear) and l.bias is not None:
                nn.init.constant_(l.bias, 0)
        elif isinstance(l, nn.LayerNorm):
            nn.init.constant_(l.bias, 0)
            nn.init.constant_(l.weight, 1.0)
    
    def forward(self, x):
        #normalise the data, input shape (B, C, H, W)
        x = self.normalise_layer(x)
        outer_shortcut = x
        x = self.patch_embed(x)
        inner_shortcut = x

        for i in range(len(self.recursive_layers)):
            x = self.recursive_layers[i](x)
            
        x=torch.add(x, inner_shortcut)
        x=self.patch_unembed(x, (self.H//self.patch_size,self.W//self.patch_size))
        x=torch.add(x, outer_shortcut)
        x=self.denormalise_layer(x)
        return x #output shape (B, C, H, W)

if __name__ == "__main__":
    
    training_image_size = 240
    patch_size = 15
    net = DeepRecursiveTransformer(96, (training_image_size, training_image_size), patch_size, 3,6, 1)

    inpu = torch.randn(4,1,240,240).cuda()
    net = net.cuda()
    outpu = net(inpu)
    print(outpu.shape)
    # summary(net.cuda(), (3, training_image_size, training_image_size))