import sys
import argparse
import matplotlib.pyplot as plt
import torch
import numpy as np
import  scipy.io as scio
import cv2
sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放

global device
from training.train_models.ACCS_csmri_and_ei_tvloss import ModelSet as OursModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_namer_90(args):
    filename = "/home/wzw/wzw/datasets/namer_mri/x_corrupted_cplx.mat"
    data = scio.loadmat(filename) # return a dict. type(data) 

    x_corrupted = data['x_corrupted']
    x_corrupted = np.abs(x_corrupted)
    x_corrupted = 1.0 * (x_corrupted - np.min(x_corrupted)) / (np.max(x_corrupted) - np.min(x_corrupted))
    x_corrupted = cv2.rotate(x_corrupted, cv2.ROTATE_90_COUNTERCLOCKWISE)
    plt.figure('Image2')
    plt.imshow(np.absolute(x_corrupted))
    plt.gray()
    plt.title('x_motion')
    plt.savefig("testing/test_namer/x_motion_90.png") #

    x_corrupted = x_corrupted[np.newaxis, np.newaxis, ...]
    x_corrupted = torch.from_numpy(x_corrupted)
    x_corrupted = x_corrupted.to(device, dtype=torch.float)
    # x_corrupted.cuda()

    modelset = OursModel(args, test=True)
    our_model = modelset.model
    our_model.eval()
    with torch.no_grad():
        res = our_model(x_corrupted)
    res = res.detach().cpu().numpy()
    image = np.squeeze(res)

    plt.figure('Image1')
    plt.imshow(np.absolute(image))
    plt.gray()
    plt.title('x_rec')
    plt.savefig("testing/test_namer/x_rec_90.png") #

def test_namer(args):
    filename = "/home/wzw/wzw/datasets/namer_mri/x_corrupted_cplx.mat"
    data = scio.loadmat(filename) # return a dict. type(data) 

    x_corrupted = data['x_corrupted']
    x_corrupted = np.abs(x_corrupted)
    x_corrupted = 1.0 * (x_corrupted - np.min(x_corrupted)) / (np.max(x_corrupted) - np.min(x_corrupted))
    # x_corrupted = cv2.rotate(x_corrupted, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # im_scale = np.max(x_corrupted)
    # x_corrupted = x_corrupted / im_scale

    plt.figure('Image2')
    plt.imshow(np.absolute(x_corrupted))
    plt.gray()
    plt.title('x_motion')
    plt.savefig("testing/test_namer/x_motion.png") #

    x_corrupted = x_corrupted[np.newaxis, np.newaxis, ...]
    x_corrupted = torch.from_numpy(x_corrupted)
    x_corrupted = x_corrupted.to(device, dtype=torch.float)
    # x_corrupted.cuda()

    modelset = OursModel(args, test=True)
    our_model = modelset.model
    our_model.eval()
    with torch.no_grad():
        res = our_model(x_corrupted)
    res = res.detach().cpu().numpy()
    image = np.squeeze(res)

    plt.figure('Image1')
    plt.imshow(np.absolute(image))
    plt.gray()
    plt.title('x_rec')
    plt.savefig("testing/test_namer/x_rec_scal.png") #


if __name__ == "__main__":
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
        default='IXI',
        help="which dataset",
        choices=['IXI', 'fastMRI'])
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default='MR_ART',
        help="which dataset",
        choices=['MR_ART'])
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='IXI',
        # choices=['ACDC', 'BrainTS', 'OAI', 'MRB'],
        choices=['IXI', 'fastMRI', 'MR_ART'],
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
    test_namer(args)