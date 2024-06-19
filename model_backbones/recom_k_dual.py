import sys

import torch
from torch import nn

sys.path.append('.')
from model_backbones.dualdomain import DualDoRecNet
from layers.recommended_k import Sampler, rfft, fft, ifft

class SequentialASLDualDo(nn.Module):
    def __init__(self, num_step, shape,  sparsity, mini_batch, args, 
                 preselect=True, line_constrained=False, preselect_num=2,
                 inputs_size=None, nclasses=None):
        super().__init__()

        self.sampler = Sampler(shape=shape, line_constrained=line_constrained, mini_batch=mini_batch)
        self.reconstructor = DualDoRecNet()
        self.num_step = num_step
        self.shape = shape
        self.preselect = preselect
        self.line_constrained = line_constrained
        
        # self.seg_net = SimpleUnet(1, nclasses)
        self.args = args
        # self.fine_recon_net = SemanticReconNet(input_nc=0, output_nc=2, out_size=inputs_size[0], n_class=nclasses+n_channles) 

        if not line_constrained:
            self.sparsity = sparsity - (preselect_num*preselect_num) / (shape[0]*shape[1]) if preselect else sparsity 
        else:
            self.sparsity = (sparsity - preselect_num / shape[1]) if preselect else sparsity

        if line_constrained:
            self.budget = self.sparsity * shape[0] 
        else:
            self.budget = self.sparsity * shape[0] * shape[1]

        self.preselect_num_one_side = preselect_num // 2

    def step_forward(self, full_kspace, pred_kspace, old_mask, step): 
        
        budget = self.budget / self.num_step  

        
        masked_kspace, new_mask = self.sampler(full_kspace, pred_kspace, old_mask, budget)
        zero_filled_recon = ifft(masked_kspace) 
        
        recon = self.reconstructor(masked_kspace, zero_filled_recon, new_mask)  
        pred_dict = {'output': recon, 'mask': new_mask, 'zero_recon': zero_filled_recon}
        return pred_dict

    def _init_mask(self, x):
        a = torch.zeros([x.shape[0], 1, self.shape[0], self.shape[1]]).to(x.device)

        if self.preselect:
            if not self.line_constrained:
                a[:, 0, :self.preselect_num_one_side, :self.preselect_num_one_side] = 1 
                a[:, 0, :self.preselect_num_one_side, -self.preselect_num_one_side:] = 1 
                a[:, 0, -self.preselect_num_one_side:, :self.preselect_num_one_side] = 1 
                a[:, 0, -self.preselect_num_one_side:, -self.preselect_num_one_side:] = 1 
            else:
                a[:, :, :, :self.preselect_num_one_side] = 1
                a[:, :, :, -self.preselect_num_one_side:] = 1
        return a

    def _init_kspace(self, f_k, initial_mask):
        u_k0 =  f_k * initial_mask
        init_img = ifft(u_k0)
        
        recon = self.reconstructor(u_k0, init_img, initial_mask)
        pred_kspace = fft(recon)
        return pred_kspace

    def forward(self, raw_img):

        full_kspace = rfft(raw_img)

        old_mask = self._init_mask(full_kspace) 
        pred_kspace = self._init_kspace(full_kspace, old_mask)  
        masks = {}      
        for i in range(self.num_step):
            pred_dict = self.step_forward(full_kspace, pred_kspace, old_mask, i)
            first_recon, new_mask, zero_recon = (pred_dict['output'], pred_dict['mask'], pred_dict['zero_recon'].detach())
            old_mask = new_mask
            pred_kspace = fft(first_recon)
            masks[i] = old_mask

        first_recon = torch.norm(first_recon, dim=1, keepdim=True)
        pred_dict = {'output': first_recon, 'mask': new_mask, 'zero_recon': zero_recon}
        return pred_dict