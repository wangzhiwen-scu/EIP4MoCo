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

import argparse
import pandas as pd
warnings.filterwarnings("ignore")
from torchvision.transforms.functional import rotate as torch_rotate


sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from data.dataset import get_h5py_unsupervised_dataset, get_h5py_supervised_dataset

from utils.train_utils import return_data_ncl_imgsize, mkdir
from utils.visdom_visualizer import VisdomLinePlotter
from testing.toolbox.test_toolbox import tensor2np, get_metrics, get_error_map, newdf2excel, dataframe_template, get_seg_oasis1_main_map

# load models

# from training.compared_models.marc import ModelSet as MARC
# from training.compared_models.cyclegan import ModelSet as cycleGAN
# from training.compared_models.bsa import ModelSet as BSA
# # from training.train_models.ACCS_csmri_and_ei_tvloss_fig1_sim import ModelSet as OursModel
# # from training.train_models.ACCS_csmri_and_ei_tvloss_prior import ModelSet as OursModel
# # from training.train_models.ACCS_csmri_and_ei_tvloss_tta import ModelSet as OursModeltta
# # from training.train_models.ACCS_csmri_and_ei_tvloss_prior_csonly import ModelSet as OursModel


from training.train_models.csmri_newei import ModelSet as OursModel
# from training.compared_models.baseline_sup import ModelSet as BaselineModel
# from training.train_models.ablation1_ei import csmri_rotcsmri_maskablation1 as Model
from training.train_models.ablation1_ei.csmri_rotcsmri_maskablation1 import ModelSet as Model

# from model_backbones.nafnet import NAFNet  # nice net, but overfitting


global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

#######################################-TEST-#########################################################

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
            # self.bbox1 = (90, 70, 90+60, 70+30)
            # self.bbox2 = (110, 150, 110+60, 150+30) 
            # self.THE_ONE = 1348

            # self.bbox1 = (81, 75, 81+80, 75+40)
            self.bbox1 = (30, 50, 30+120, 50+60)
            self.bbox2 = (90, 180, 90+80, 180+40) 
            self.THE_ONE = 1467

def loop_test():

    cross_motion = ['ixi_t1_linear_moderate', 'ixi_t1_nonlinear_moderate', 'ixi_t1_sudden_moderate', \
                    'ixi_t1_singleshot_moderate', 'ixi_t1_periodic_slight_rl', 'ixi_t1_periodic_moderate', 'ixi_t1_periodic_heavy']
    one_motion = ['ixi_t1_periodic_slight_rl']

    for motion in one_motion:
        test(motion)

def test(ourtta):

    file_name = os.path.basename(__file__)
    filename, _suffix = os.path.splitext(file_name)
    plotter = VisdomLinePlotter(env_name='test')
    
    namespace = NameSpace(figs=2)
    # if args.dataset_name:
        # data_name = args.supervised_dataset_name
    # else:
    data_name = 'MR_ART'

    conditional_filename =  './results/ablation/{}/{}_{}_{}'.format(filename, ourtta, args.direction, args.headmotion) 
    args.PREFIX = conditional_filename
    mkdir(conditional_filename)

    _traindataset, val_loader = get_h5py_supervised_dataset(ourtta, test=True)
    dataloader = val_loader

    args.ablation_which = 'ablation2'
  

    args.ablation_var = 'equidistant'
    modelset = Model(args, test=True)
    model_equidistant = modelset.model  
    model_equidistant.eval()

    args.ablation_var = 'gaussian'
    modelset = Model(args, test=True)
    model_gaussian = modelset.model  
    model_gaussian.eval()

    # args.supervised_dataset_name = 'ixi_t1_periodic_slight'
    args.stage = '1'
    modelset = OursModel(args, test=True)
    model_random = modelset.model
    model_random.eval()


    imgs = {}
    masks = {}
    segs = {}
    # df = {'Corrupted': None, 'MARC': None, 'BSA': None, 'OursPrior': None, 'Ours':None}
    df = {'equidistant': None, 'gaussian': None, 'random': None}

    template = dataframe_template(dataset_name=data_name)
    
    # initialize
    for key in df.keys():
        df[key] = pd.DataFrame(template)

    NUM_SLICE = 0
    # patient_namelists = data_gallary[data_name](test=isTestdata,  args=args)[0] # 把每个slice的名称存起来。好消息：每个文件名自带患者信息。
    with torch.no_grad():
        for x_iter in dataloader:
            still, with_motion, _shotname = x_iter
            inputs = with_motion.to(device, dtype=torch.float)
            still = still.to(device, dtype=torch.float)

            # if ourtta == 'ixi_t1_periodic_slight_rl':
            #     inputs = torch_rotate(inputs, -90)
            #     still = torch_rotate(still, -90)

            shotname = _shotname[0]  # dataloader makes string '1' to list ('1',)
            imgs['gt'] = tensor2np(still)

            # imgs['Corrupted'] = tensor2np(with_motion)

            pred_results = model_equidistant(inputs)
            imgs['equidistant'] = tensor2np(pred_results)

            pred_results = model_gaussian(inputs)
            imgs['gaussian'] = tensor2np(pred_results)

            pred_results = model_random(inputs)
            imgs['random'] = tensor2np(pred_results)


            get_metrics(args, imgs, df, shotname, NUM_SLICE)  # ? update psnrs and ssims , and get the current metrics - range -1~2000
            # get_error_map(args, imgs, NUM_SLICE, df, data_name, shotname) # ! save error map
            # get_seg_oasis1_main_map(args, imgs, NUM_SLICE, df, data_name, shotname, namespace, dpi=200)

            # get_seg_oasis1_main_map(args, imgs, NUM_SLICE, df, data_name, shotname, namespace, dpi=200, oneline=True) # for one line.

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
    parser.add_argument(
        "--ablation_which",
        type=str,
        default=None,
        help="mask_ablation choose which ablation")
    parser.add_argument(
        "--ablation_var",
        type=str,
        default=None,
        help="mask_ablation choose which control vars of ablation1,2,3")
    args = parser.parse_args()
    torch.cuda.set_device(args.cuda)
    # test()
    loop_test()


    # TODO: For seg only. should be writen in to `test(arg_data=dataset)`
    # TODO: {DATASET_TRAJEC}, MRB_Radial