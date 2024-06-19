"""
    Mask Learning Module.
    Pytorch version.
    By WZW.
"""
import os
import sys
import torch
import warnings

sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from training.train_models.ACCS_csmri_and_ei_tvloss_prior_restormer import ModelSet as Restormer
from training.train_models.ACCS_csmri_and_ei_tvloss_prior_MSTPP import ModelSet as MSTPP
from training.train_models.ACCS_csmri_and_ei_tvloss_prior_unet import ModelSet as Unet
# from model_backbones.recon_net import Unet

from training.train_models.ACCS_csmri_and_ei_tvloss_prior_stripformer import ModelSet as Stripformer  # nice transformer. but slow.
# from model_backbones.drt import DeepRecursiveTransformer #

from training.train_models.csmri_newei_spatial import ModelSet as OursModel

# from models.helper import ssim, psnr2
import numpy as np

import argparse

from utils.train_utils import return_data_ncl_imgsize
from ptflops import get_model_complexity_info


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
#######################################-TEST-#########################################################
import subprocess





def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def test():
    # module = ''
    # args.desired_sparsity = args.rate
    # args.nclasses, inputs_size = return_data_ncl_imgsize(args.dataset_name)
    # asl_nclasses = args.nclasses+ 2
    # args.nclasses = 1
    # if args.maskType == '1D':
    #     traj_type = 'cartesian'
    # elif args.maskType == '2D':
    #     traj_type = 'random'
        
    # args.mask = traj_type
    # acc = int(args.desired_sparsity * 100)

    # ckpt = './model_zoo/tab1/{}/asl_seqmdrecnet_bg_step3_1_{}_{}_{}.pth'.format(args.dataset_name, module, str(args.rate), args.maskType)
    # args.segpath =  "/home/labuser1/wzw/asl/model_zoo/pretrained_seg/{}_{}_mixedbackground.pth".format(args.dataset_name, asl_nclasses)
    # args.save_path = ckpt

    modelset = Unet(args, test=True)
    model_unet = modelset.model  
    model_unet.eval()

    # args.ablation_var = '30'
    modelset = Restormer(args, test=True)
    model_restormer = modelset.model  
    model_restormer.eval()

    # args.ablation_var = '40'
    modelset = Stripformer(args, test=True)
    model_stripformer = modelset.model  
    model_stripformer.eval()

    # args.ablation_var = '50'
    modelset = MSTPP(args, test=True)
    model_mstpp = modelset.model  
    model_mstpp.eval()

    # args.supervised_dataset_name = 'ixi_t1_periodic_slight'
    args.stage = '1'
    modelset = OursModel(args, test=True)
    model_our = modelset.model
    model_our.eval()

    model = model_our


    # csmri1 = CSMRIUnet(args.rate, args.mask, inputs_size).to(device, dtype=torch.float)

    # create_dual_model = CSMRI2(args, acc=acc, mask_type = traj_type )
    # csmri2 = create_dual_model.model.to(device, dtype=torch.float)
    

    # csmtl = CSMTLNet(args=args, nclasses=args.nclasses, inputs_size=inputs_size).to(device, dtype=torch.float)

    # csl_unet = SequentialUnet(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=args.rate, preselect_num=2, mini_batch=args.batch_size, 
                                        # reconstructor_name='ista') 
    # aslstage3 = SequentialASL(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=args.rate, 
                                        # preselect_num=2, mini_batch=args.batch_size, inputs_size=inputs_size, nclasses=asl_nclasses, args=args)  

    # csl_mdrec = CSLRec(num_step=3, shape=[240,240], preselect=True, line_constrained=False, sparsity=args.rate, preselect_num=2, mini_batch=args.batch_size, 
                                # reconstructor_name='ista')

    # csl_mdrec = ASLStage1(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=args.rate, 
                            # preselect_num=2, mini_batch=args.batch_size, inputs_size=inputs_size, nclasses=args.nclasses, args=args).to(device, dtype=torch.float)

    # aslstage1 = ASLStage1(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=args.rate, 
                            # preselect_num=2, mini_batch=args.batch_size, inputs_size=inputs_size, nclasses=asl_nclasses, args=args).to(device, dtype=torch.float)

    # aslstage2 = ASLStage2(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=args.rate, 
    #                         preselect_num=2, mini_batch=args.batch_size, inputs_size=inputs_size, nclasses=asl_nclasses, args=args).to(device, dtype=torch.float)

    # aslstage3 = ASLStage3(num_step=3, shape=[240,240], preselect=True, line_constrained=True, sparsity=args.rate, 
                            # preselect_num=2, mini_batch=args.batch_size, inputs_size=inputs_size, nclasses=asl_nclasses, args=args).to(device, dtype=torch.float)


    # https://deci.ai/blog/measure-inference-time-deep-neural-networks/

    # model = aslstage2.to(device, dtype=torch.float)

    dummy_input = torch.randn(1, 1, 240, 240, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    
    for _ in range(10):
        _ = model(dummy_input)

    # print(torch.cuda.mem_get_info(device=0)) # torch 1.12.0
    # memory_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    # print(torch.cuda.reset_peak_memory_stats())
    # memory_usage = get_gpu_memory_map()
    # print(memory_usage)
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))


    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(mean_syn)

        # macs, params = get_model_complexity_info(csmri1, (1, 240, 240), as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print("csmri1")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))


        # macs, params = get_model_complexity_info(csmtl, (1, 240, 240), as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print("csmtl")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # macs, params = get_model_complexity_info(aslstage1, (1, 240, 240), as_strings=True,
                                                # print_per_layer_stat=False, verbose=True)
        # print("aslstage1")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # macs, params = get_model_complexity_info(csl_unet, (1, 240, 240), as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print("csl_unet")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # macs, params = get_model_complexity_info(aslstage3, (1, 240, 240), as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print("aslstage3")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # macs, params = get_model_complexity_info(csl_mdrec, (1, 240, 240), as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print("csl_mdrec")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        # macs, params = get_model_complexity_info(csmri2, (2, 240, 240), as_strings=True,
        #                                         print_per_layer_stat=False, verbose=True)
        # print("csmri2")
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

        macs, params = get_model_complexity_info(model, (1, 240, 240), as_strings=True,
                                                print_per_layer_stat=False, verbose=True)
        print("model")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    
        # pred_results = dualmodel(inputs)

        # pred_results = liumodel(inputs)

        # pred_results = recon_model(inputs)

        # pred_results = comb_model(inputs)

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
    test()


    # TODO: For seg only. should be writen in to `test(arg_data=dataset)`
    # TODO: {DATASET_TRAJEC}, MRB_Radial