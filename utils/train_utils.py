import os
import sys
import time
import random

import torch
import numpy as np

def return_data_ncl_imgsize(dataname):
    if dataname == 'MRB':
        # nclasses = 7
        # inputs_size = [240, 240]
        pass
    elif dataname == 'OAI': # knee
        nclasses = 6
        inputs_size = [240, 240]
    elif dataname == 'ACDC': # cine MRI
        nclasses = 3
        inputs_size = [240, 240]
    elif dataname == 'BrainTS': # cine MRI
        nclasses = 3
        inputs_size = [192, 192]
    elif dataname == 'MICCAI': # cine MRI
        nclasses = 33
        inputs_size = [240, 240]
    elif dataname == 'OASI1_MRB': # cine MRI
        nclasses = 3
        inputs_size = [240, 240]
    elif dataname == 'Prostate': # cine MRI
        nclasses = 2
        inputs_size = [240, 240]
    return nclasses, inputs_size


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def get_filePath_fileName_fileExt(fileUrl):
    """获取文件路径， 文件名， 后缀名
    :param fileUrl:
    :return
    """
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension

def get_weights(ckpt, model):
    pre_trained_model = torch.load(ckpt)
    new = list(pre_trained_model.items())
    length = len(new)
    my_model_kvpair = model.state_dict()
    count = 0
    for key, value in my_model_kvpair.items():
        layer_name, weights = new[count]
        my_model_kvpair[key] = weights
        count += 1
        # break
        if count == length:
            break
    model.load_state_dict(my_model_kvpair, False)  # (my_model_kvpair, False)

def update_partial_dict(pretrained_dict, model):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)



def update_mask_dict(pretrained_dict, model):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in 'mask_layer.layer_probmask.weight'}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def torch_binarize(k_data, desired_sparsity):
    batch, channel, x_h, x_w = k_data.shape
    tk = int(desired_sparsity*x_h*x_w)+1
    k_data = k_data.reshape(batch, channel, x_h*x_w, 1)
    values, indices = torch.topk(k_data, tk, dim=2)
    k_data_binary =  (k_data >= torch.min(values))
    k_data_binary = k_data_binary.reshape(batch, channel, x_h, x_w).float()
    return k_data_binary

class Logger(object):
    def __init__(self, stream=sys.stdout):
        output_dir = "log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print( "---  new folder...  ---")
		print("---  OK  ---")
 
	else:
		print("---  There is this folder!  ---")