import torch.nn as nn
from layers.dota_mri_layer import fftshift2, GenConvBlock, GenFcBlock, ifft1, AbsLayer, DataConsist
from torch import cat, reshape
    
class DOTA(nn.Module):
    def __init__(self, opt):
        super(DOTA, self).__init__()

        FC_blocks = []
        conv_blocks_K = []        
        conv_blocks_I = []
        
        FC_blocks.append(GenFcBlock([opt.im_height * opt.in_ch, 1024, 1024, opt.im_height * opt.out_ch]))
        conv_blocks_K.append(GenConvBlock(opt.k, opt.in_ch, opt.out_ch, opt.fm))
        conv_blocks_I.append(GenConvBlock(opt.i, opt.in_ch, opt.out_ch, opt.fm))

        self.FC_blocks     = nn.ModuleList(FC_blocks)
        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        
        self.n_kx   = opt.im_width
        self.n_ky   = opt.im_height
        self.in_ch  = opt.in_ch
        self.out_ch = opt.out_ch
        self.abs_layer = opt.abs_layer
        if opt.dota_dc != 0:
            self.dc = True
        else:
            self.dc = False

    def forward(self, kspace_us, mask):
        rec = kspace_us
        rec = fftshift2(rec)
        rec = self.conv_blocks_K[0](rec)
        
        rec = ifft1(rec, 1)
        
        out_lines = []
        for i in range(self.n_kx):
            in_line  = reshape(rec[:, :, :, i], (-1, self.in_ch * self.n_ky))
            out_line = self.FC_blocks[0](in_line)
            out_line = reshape(out_line, (-1, self.out_ch, self.n_ky, 1))
            out_lines.append(out_line)

        rec = cat(out_lines, 3)
        rec = rec + self.conv_blocks_I[0](rec)
        
        if self.dc:
            rec = DataConsist(rec, kspace_us, mask)
        
        if self.abs_layer:
            rec = AbsLayer(rec)

        return rec