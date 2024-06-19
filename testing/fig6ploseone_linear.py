"""
    Mask Learning Module.
    Pytorch version.
    By WZW.
"""
import os
import sys
import time
import torch
import warnings

# from models.helper import ssim, psnr2
import numpy as np
from matplotlib.image import imsave
from mpl_toolkits.axes_grid1 import AxesGrid
from copy import deepcopy
import cv2
import scipy.io as sio


from matplotlib import pyplot as plt

import argparse
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from data.dataset import get_h5py_unsupervised_dataset, get_h5py_supervised_dataset

from utils.train_utils import return_data_ncl_imgsize, mkdir
from utils.visdom_visualizer import VisdomLinePlotter
from testing.toolbox.test_toolbox import tensor2np, get_metrics, get_error_map, newdf2excel, dataframe_template, get_seg_oasis1_main_map, get_seg_ablation_map, compared_PE_line
from utils.split_bregman import excute_split_bregman
# load models

from training.compared_models.marc import ModelSet as MARC
# from training.compared_models.baseline_sup import ModelSet as MARC


from training.compared_models.cyclegan import ModelSet as cycleGAN
from training.compared_models.bsa import ModelSet as BSA
# from training.train_models.ACCS_csmri_and_ei_tvloss_fig1_sim import ModelSet as OursModel
# from training.train_models.ACCS_csmri_and_ei_tvloss_prior import ModelSet as OursModel
# from training.train_models.ACCS_csmri_and_ei_tvloss_tta import ModelSet as OursModel
# from training.train_models.ACCS_csmri_and_ei_tvloss_prior_bezier import ModelSet as OursModel
# from training.train_models.csmri_ei import ModelSet as OursModel
from training.train_models.csmri_newei_spatial import ModelSet as OursModel

from training.compared_models.baseline_sup import ModelSet as BaselineModel


# split bregman
import matlab.engine

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#######################################-TEST-#########################################################
def NormalizeData(data, datarange=1):
    return datarange*(data - np.min(data)) / (np.max(data) - np.min(data))

class NameSpace(object):
    def __init__(self, figs):
        #    x1,y1 -----(x2, y1)
        #    |          |
        #    |          |
        #    |          |
        # (x1, y2)-------x2,y2
        if figs == 2:
            self.bbox1 = (128, 100, 128+30, 100+30)
            self.bbox2 = (95, 170, 95+30, 170+30) 
            self.THE_ONE = 1 # 1106
        elif figs == 3:

            self.bbox1 = (30, 50, 30+120, 50+60)
            self.bbox2 = (90, 180, 90+80, 180+40) 
            self.THE_ONE = 1467
        elif figs == 4:
            self.bbox1 = (100, 100+40, 190, 190+40)

def test():

    file_name = os.path.basename(__file__)
    filename, _suffix = os.path.splitext(file_name)
    plotter = VisdomLinePlotter(env_name='test')
    
    namespace = NameSpace(figs=4)
    # if args.dataset_name:
        # data_name = args.supervised_dataset_name
    # else:
    data_name = 'MR_ART'

    conditional_filename =  './results/{}/{}_{}_{}'.format(filename, data_name, args.direction, args.headmotion) 
    args.PREFIX = conditional_filename
    mkdir(conditional_filename)

    args.supervised_dataset_name = 'ixi_t1_periodic_slight'
    # ourtta = 'ixi_t1_periodic_moderate'

    ourtta, theone = 'ixi_t1_linear_moderate', 753
    # ourtta, theone = 'ixi_t1_nonlinear_moderate', 410
    # ourtta, theone = 'ixi_t1_sudden_moderate', 228
    # ourtta, theone = 'ixi_t1_singleshot_moderate', 419
    # ourtta, theone = 'ixi_t1_singleshot_moderate', 400 # RL have special ❌
    # ourtta, theone = 'ixi_t1_periodic_moderate', 513
    # ourtta, theone = 'ixi_t1_periodic_heavy', 237


    _traindataset, val_loader = get_h5py_supervised_dataset(ourtta, test=True)
    dataloader = val_loader

    # model MARC
    modelset = MARC(args, test=True)
    marc_model = modelset.model  
    marc_model.eval()

    args.supervised_dataset_name = 'ixi_t1_periodic_slight'
    args.stage = '1'
    modelset = BaselineModel(args, test=True)
    pe_model = modelset.model
    pe_model.eval()


    # model cycleGAN med v2
    modelset = cycleGAN(args, test=True)
    cycle_model = modelset.model  
    cycle_model.eval()

    # model BSA
    args.stage = '1'
    modelset = BSA(args, test=True)
    bsa_model = modelset.model     
    bsa_model.eval()

    
    args.stage = '1'
    modelset = OursModel(args, test=True)
    our_model = modelset.model
    our_model.eval()


    args.supervised_dataset_name = ourtta
    args.stage = '1'
    modelset = BaselineModel(args, test=True)
    supervised_model = modelset.model
    supervised_model.eval()


    imgs = {}
    masks = {}
    segs = {}
    # df = {'Corrupted': None, 'MARC': None, 'BSA': None, 'OursPrior': None, 'Ours':None}
    # df = {'Corrupted': None, 'MARC': None, 'cycleGAN': None, 'BSA': None, 'Ours':None}
    df = {'Corrupted': None, 'MARC': None, 'PE': None, 'cycleGAN': None, 'BSA': None, 'Ours':None, 'Supervised':None}

    template = dataframe_template(dataset_name=data_name)
    # initialize
    for key in df.keys():
        df[key] = pd.DataFrame(template)

    keys_list = list(df.keys())
    keys_list.insert(0, 'gt')

    NUM_SLICE = 0
    # patient_namelists = data_gallary[data_name](test=isTestdata,  args=args)[0] # 把每个slice的名称存起来。好消息：每个文件名自带患者信息。

    eng = matlab.engine.start_matlab()
    eng.cd(r'./utils/matlab_splitbregman', nargout=0)

    with torch.no_grad():
        for x_iter in dataloader:
            still, with_motion, _shotname = x_iter
            inputs = with_motion.to(device, dtype=torch.float)
            still = still.to(device, dtype=torch.float)

            B,C,H,W = still.shape

            shotname = _shotname[0]  # dataloader makes string '1' to list ('1',)

            # if NUM_SLICE != theone:
                # print(NUM_SLICE)
                # NUM_SLICE += 1
                # continue


            imgs['gt'] = tensor2np(still)

            imgs['Corrupted'] = tensor2np(with_motion)

            predicted_residuals = marc_model(inputs)
            recon_baseline = inputs - predicted_residuals
            np_img= tensor2np(recon_baseline)
            imgs['MARC'] = np_img

            # recon_baseline = marc_model(inputs)
            # np_img= tensor2np(recon_baseline)

            # predicted_residuals = pe_model(inputs)
            # np_img= tensor2np(predicted_residuals)

            mask_from_pe_line, _sub_k = compared_PE_line(imgs['gt'], np_img, random_mask=True)
            np_img = np_img.astype(float)
            mask_from_pe_line = mask_from_pe_line.astype(float)
            rec_img = eng.mysplitbregman(np_img, mask_from_pe_line)
            # rec_img = eng.mysplitbregman(imgs['Corrupted'], mask_from_pe_line)
            rec_img = np.array(rec_img)
            # rec_img = NormalizeData(rec_img)
            rec_img = np.abs(rec_img)
            imgs['PE'] = rec_img

            # print(shotname + '  '+ str(inputs.shape))

            pred_results = cycle_model(inputs)
            imgs['cycleGAN'] = tensor2np(pred_results)
            recon_bsa = bsa_model(inputs)
            imgs['BSA'] = tensor2np(recon_bsa)


            # pred_results = cycle_model(inputs)
            # imgs['OursPrior'] = tensor2np(pred_results)

            recon_bsa = our_model(inputs)
            imgs['Ours'] = tensor2np(recon_bsa)

            recon_sup = supervised_model(inputs)
            imgs['Supervised'] = tensor2np(recon_sup)

            get_metrics(args, imgs, df, shotname, NUM_SLICE)  # ? update psnrs and ssims , and get the current metrics - range -1~2000
            # get_error_map(args, imgs, NUM_SLICE, df, data_name, shotname) # ! save error map
            # get_seg_oasis1_main_map(args, imgs, NUM_SLICE, df, data_name, shotname, namespace, dpi=200)
            # get_seg_oasis1_main_map(args, imgs, NUM_SLICE, df, data_name, shotname, namespace, dpi=200, oneline=True) # for one line.
            # get_seg_ablation_map(args, imgs, NUM_SLICE, df, data_name, shotname, keys_list, have_title=False, namespace=namespace, dpi=200, oneline=True) # for one line.

            for dfkey in df.keys():
                if dfkey in ['shotname', 'slice']:
                    continue
                else:
                    print('{}:'.format(dfkey))
                    
                    print(df[dfkey].mean())

            NUM_SLICE += 1
            # break
        
        excel_filename_pre = conditional_filename + time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
        excel_filename = excel_filename_pre + '.xlsx' 


        newdf2excel(excel_filename, df)
        print('Saving {}'.format(excel_filename))
        # toolbox.read_excel_and_plot(excel_filename, excel_filename_pre, 'png')
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