"""
    Mask Learning Module.
    Pytorch version.
    By WZW.
"""
import os
import sys
import torch
import warnings

import numpy as np
# from models.helper import ssim, psnr2
import matplotlib.pyplot as plt
import cv2

import argparse
warnings.filterwarnings("ignore")

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放

from utils.train_utils import return_data_ncl_imgsize, mkdir
from utils.visdom_visualizer import VisdomLinePlotter
from data.toolbox import OASI1_MRB, IXI_T1_motion, MR_ART

from model_backbones.recon_net import Unet, Hand_Tailed_Mask_Layer
from training.train_models.ACCS_csmri_and_ei_tvloss_prior_unet import ModelSet as UNet


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#######################################-TEST-#########################################################

def get_240img4realmotion(input_img):
    
    input_img = cv2.rotate(input_img, cv2.ROTATE_180)

    offfset256to240 = input_img.shape[0]  - 240
    input_img = input_img[offfset256to240-1:-1, :]
    input_img = input_img[:,:, np.newaxis] # (256,192) -> (256,192,1)
    x_offset = input_img.shape[1] - 240
    y_offset = input_img.shape[1] - 240
    offset = (int(abs(x_offset/2)), int(abs(y_offset/2)))
    npad = ((0, 0), offset, (0, 0)) # (240,180,1) -> (240,240,1)
    output = np.pad(input_img, pad_width=npad, mode='constant', constant_values=0)  # (z, x, y) 
    # input_img = np.square(input_img) # (240,180) -> (240,180,1)
    # output = cv2.resize(output, [240, 240])
    
    output = np.squeeze(output, axis=2)
    # output = cv2.rotate(output, cv2.ROTATE_180)
    
    return output

def preprosesing(img, rotate=False, resize=False):

    img = img.astype(np.float32)
    img = 1.0 * (img - np.min(img)) / (np.max(img) - np.min(img))

    if rotate:
        img = cv2.resize(img, [240, 240])

    if resize:
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img

def subsampling(full_img_numpy, sampnet):
    full_img_tensor = torch.from_numpy(full_img_numpy).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
    output = sampnet(full_img_tensor)
    zero_filling_tensor = output['complex_abs']
    zero_filling_numpy = zero_filling_tensor.squeeze().detach().cpu().numpy()
    return zero_filling_numpy

def deepunet(full_img_numpy, unet):
    full_img_tensor = torch.from_numpy(full_img_numpy).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float)
    output_tensor = unet(full_img_tensor)

    output_numpy = output_tensor.squeeze().detach().cpu().numpy()

    return output_numpy

def normal_gray(still):
    # still = 1.0 * (still - np.min(still)) / (np.max(still) - np.min(still))
    still = still / np.max(still)
    return still

def plot_images(images_dict, rows, cols):
    # images =  lists_dict  
    keys = ['t1', 't2', 'pd', 'real']

    # fig, axes = plt.subplots(rows, cols, figsize=(8, 6))
    fig, axes = plt.subplots(rows, cols, figsize=(7.8, 8))
    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols

        if row == 0:  # First row
            img = images_dict[keys[col]][0]
            cmap = 'gray'
        elif row == 1:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images_dict[keys[col]][1]
            cmap = 'gray'
        elif row == 2:  # Last row
            img = images_dict[keys[col]][2]
            cmap = 'gray'
        elif row == 3:  # Last row
            img = images_dict[keys[col]][3]
            cmap = 'gray'

        else:
            ax.axis('off')
            continue
        ax.imshow(img, cmap=cmap)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    save_path = './testing/article_fig1/fig1.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)


def test():

    file_name = os.path.basename(__file__)
    filename, _suffix = os.path.splitext(file_name)
    plotter = VisdomLinePlotter(env_name='test')
    
    # if args.dataset_name:
        # data_name = args.supervised_dataset_name
    # else:
    data_name = 'MR_ART'

    conditional_filename =  './results/{}/{}_{}_{}'.format(filename, data_name, args.direction, args.headmotion) 
    args.PREFIX = conditional_filename
    mkdir(conditional_filename)

    # path
    t1_example = {'path': './data/datasets/IXI_T1_motion/ixi_t1_periodic_moderate/testing-h5py/motion_paried_IXI324-Guys-0922-T1.nii.h5',
                  'slice': 5}
    t2_example = {'path': './data/datasets/IXI_T1_motion/ixi_t2_periodic_moderate/testing-h5py/motion_paried_IXI019-Guys-0702-T2.nii.h5',
                  'slice': 5}
    pd_example = {'path': './data/datasets/IXI_T1_motion/ixi_pd_periodic_moderate/testing-h5py/motion_paried_IXI041-Guys-0706-PD.nii.h5',
                  'slice': 5}
    
    real_example = {'path': './data/datasets/MR-ART/testing-h5py/motion_triple_sub-107738_acq-headmotion1_T1w.nii.h5',
                  'slice': 9}

    # read path

    sample_net = Hand_Tailed_Mask_Layer(0.33, 'cartesian', (240, 240), resize_size=[240,240]).to(device, dtype=torch.float)

    args.stage = '1'
    modelset = UNet(args, test=True)
    unet = modelset.model
    unet.eval()

    fname, which_slice = t1_example['path'], t1_example["slice"]
    still, with_motion = IXI_T1_motion.getslicefromh5py(fname, which_slice) # for t1, t2, pd  
    still, with_motion  = preprosesing(still, rotate=False, resize=False), preprosesing(with_motion, rotate=False, resize=False)
    zero_filling = subsampling(still, sample_net)

    motion_correction = deepunet(with_motion, unet)
    t1wi_lists = [still, with_motion, zero_filling, motion_correction]

    fname, which_slice = t2_example['path'], t2_example["slice"]
    still, with_motion = IXI_T1_motion.getslicefromh5py(fname, which_slice) # for t1, t2, pd  
    still, with_motion  = preprosesing(still, rotate=True, resize=True), preprosesing(with_motion, rotate=True, resize=True)
    zero_filling = subsampling(still, sample_net)
    motion_correction = deepunet(with_motion, unet)
    t2wi_lists = [still, with_motion, zero_filling, motion_correction]

    fname, which_slice = pd_example['path'], pd_example["slice"]
    still, with_motion = IXI_T1_motion.getslicefromh5py(fname, which_slice) # for t1, t2, pd  
    still, with_motion  = preprosesing(still, rotate=True, resize=True), preprosesing(with_motion, rotate=True, resize=True)
    zero_filling = subsampling(still, sample_net)
    motion_correction = deepunet(with_motion, unet)
    pdwi_lists = [still, with_motion, zero_filling, motion_correction]

    fname, which_slice = real_example['path'], real_example["slice"]
    headmotion1, headmotion2, standard = MR_ART.getslicefromh5py(fname, which_slice)
    # still, with_motion = IXI_T1_motion.getslicefromh5py(fname, which_slice) # for t1, t2, pd  
    still, with_motion = standard, headmotion2
    still, with_motion  = get_240img4realmotion(still), get_240img4realmotion(with_motion)
    zero_filling = subsampling(still, sample_net)
    motion_correction = deepunet(with_motion, unet)
    realmotion_lists = [still, with_motion, zero_filling, motion_correction]


    lists_dict = {}
    lists_dict['t1'] = t1wi_lists
    lists_dict['t2'] = t2wi_lists
    lists_dict['pd'] = pdwi_lists
    lists_dict['real'] = realmotion_lists

    plot_images(lists_dict, 4, 4)
    # re-arrange it.
    



#-----------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", type=int, default=1, choices=[0,1], help="The learning rate") # 
    parser.add_argument("--lr", type=float, default=5e-4, help="The learning rate") # 5e-4, 5e-5, 5e-6
    parser.add_argument(
        "--batch_size", type=int, default=12, help="The batch size")
    parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        choices=[0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.20, 0.25],  # cartesian_0.1 is bad;
        help="The undersampling rate")
   
    parser.add_argument(
        "--mask",
        type=str,
        default="random",
        choices=["cartesian", "radial", "random"],
        help="The type of mask")
              
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="The GPU device")
    parser.add_argument(
        "--bn",
        type=bool,
        default=False,
        choices=[True, False],
        help="Is there the batchnormalize")
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="which model")
    parser.add_argument(
        "--supervised_dataset_name",
        type=str,
        default='ixi_t1_periodic_slight',
        help="which dataset",
        # choices=['IXI', 'fastMRI']
        )
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default='MR_ART',
        help="which dataset",
        # choices=['MR_ART']
        )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='ixi_t1_periodic_slight',
        # choices=['ACDC', 'BrainTS', 'OAI', 'MRB'],
        # choices=['IXI', 'fastMRI', 'MR_ART'],
        help="which dataset")
    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        choices=[True],
        help="If this program is a test program.")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="2D",
        choices=['1D', '2D'])
    parser.add_argument(
        "--nclasses",
        type=int,
        default=None,
        choices=[0,1,2,3,4,5,6,7,8],
        help="nclasses of segmentation")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--PREFIX",
        type=str,
        default="./testing/")
    parser.add_argument(
        "--stage",
        type=str,
        default='3',
        choices=['1','2', '3'],
        help="1: coarse rec net; 2: fixed coarse then train seg; 3: fixed coarse and seg, then train fine; 4: all finetune.")
    parser.add_argument(
        "--stage1_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--stage2_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--stage3_epochs", type=int, default=800, help="The weighted parameter") #
    parser.add_argument(
        "--load_model",
        type=int,
        default=1,
        choices=[0,1],
        help="reload last parameter?")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None)
    parser.add_argument(
        "--maskType",
        type=str,
        default='2D',
        choices=['1D', '2D'])
    parser.add_argument(
        "--direction",
        type=str,
        default='Axial',
        choices=['Axial', 'Coronal', 'Sagittal'])
    parser.add_argument(
        "--headmotion",
        type=str,
        default=None,
        choices=['Slight', 'Heavy'])

    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    test()


    # TODO: For seg only. should be writen in to `test(arg_data=dataset)`
    # TODO: {DATASET_TRAJEC}, MRB_Radial