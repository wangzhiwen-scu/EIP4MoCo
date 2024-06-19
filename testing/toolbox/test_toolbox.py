import sys
import os

import torch
from torch.autograd import Variable
import numpy as np
from numpy.fft import fftshift, ifftshift
import scipy.io as sio
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, normalized_root_mse as compare_nrmse
# from scipy import signaltonoise

# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageFolder
# from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import zoom

from copy import deepcopy
import cv2
from matplotlib.image import imsave
from utils.train_metrics import dice_coef
import pandas as pd
# from PIL import Image
# from matplotlib.gridspec import GridSpec
# from matplotlib import gridspec
import lpips
from torchvision.transforms import ToTensor


sys.path.append('.') # 自己的py文件所在的文件夹路径，该放__init__.py的就放
from layers.mask_layer import Mask_Fixed_CartesianLayer, Mask_Fixed_Layer
from utils.train_utils import return_data_ncl_imgsize
from utils.motion_metrics import calculate_snr, calculate_efc, calculate_cjv

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Load the pre-trained Inception v3 network
# inception_model = torch.hub.load('pytorch/vision', 'inception_v3', pretrained=True, transform_input=False)
# inception_model.eval()

# Load the pre-trained LPIPS model
# loss_lpips = lpips.LPIPS(net='vgg', version='0.1')
# loss_lpips.eval()

def fakedataloder(input, mask_torch):

    label_img = input[0, 0, ...].detach().cpu().numpy()
    # label_img = cv2.resize(label_img, (256, 256))
    # min = np.min(label_img)
    # max = np.max(label_img)

    # label_img = (label_img - min) / (max - min) * 255.0
    full_img = torch.zeros(*label_img.shape, 2)
    full_img[:, :, 0] = torch.from_numpy(label_img.real)
    full_img[:, :, 1] = torch.from_numpy(label_img.imag)
    full_img = full_img.cuda()

    # full_k = torch.fft(full_img, 2, normalized=True)
    full_k = torch.fft(full_img, 2, normalized=False)
    tmp = full_k.permute(2, 0, 1)
    under_k = tmp * mask_torch

    tmp = under_k.permute(1, 2, 0)
    under_img = torch.ifft(tmp, 2, normalized=False)

    full_k = full_k.permute(2, 0, 1)
    full_img = full_img.permute(2, 0, 1)
    under_img = under_img.permute(2, 0, 1)
    return under_img, under_k, full_img, full_k

def createinput(f_img, mask_torch):
    """ f_img.shape = (1, 1, H, W)
    """
    under_img, under_k, full_img, full_k = fakedataloder(f_img, mask_torch)
    under_img = torch.unsqueeze(under_img, dim=0)
    under_k = torch.unsqueeze(under_k, dim=0)
    full_img = torch.unsqueeze(full_img, dim=0)
    full_k = torch.unsqueeze(full_k, dim=0)

    u_img = Variable(under_img, requires_grad=False).cuda()
    u_k = Variable(under_k, requires_grad=False).cuda()
    img = Variable(full_img, requires_grad=False).cuda()
    k = Variable(full_k, requires_grad=False).cuda()
    return u_img, u_k, img, k

def mergeMultiLabelToOne(segs):
    """Inputs: [n, seg_x, seg_y], n: how many segmentation.

    Args:
        segs (_type_): _description_
    """
    new_seg = np.zeros((segs.shape[1], segs.shape[2]))
    for i in range(segs.shape[0]):
        new_seg[segs[i]==1] =1
    return new_seg

def mask_find_bboxs(mask):
    """
    https://zhuanlan.zhihu.com/p/59486758
    mask 
    bin_uint8 = (binarized * 255).astype(np.uint8)
    You need to convert the image data type to uint8
    """
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8) # connectivity参数的默认值为8
    stats = stats[stats[:,4].argsort()]
    return stats[:-1] # 排除最外层的连通图

def get_roi(imgs_gt, imgs_key, segs_gt):
    """https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
    It is a straight rectangle, it doesn’t consider the rotation of the object. So area of the bounding rectangle won’t be minimum. 
    It is found by the function cv2.boundingRect().
    Let (x,y) be the top-left coordinate of the rectangle and (w,h) be its width and height.

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    https://zhuanlan.zhihu.com/p/59486758
    https://stackoverflow.com/questions/9084609/how-to-copy-a-image-region-using-opencv-in-python
    ROI = image[y1:y2, x1:x2]

    -------------------------------------------
    |                                         | 
    |    (x1, y1)                             |
    |      ------------------------           |
    |      |                      |           |
    |      |                      |           | 
    |      |         ROI          |           |  
    |      |                      |           |   
    |      |                      |           |   
    |      |                      |           |       
    |      ------------------------           |   
    |                           (x2, y2)      |    
    |                                         |             
    |                                         |             
    |                                         |             
    -------------------------------------------
    Args:
        imgs_gt (_type_): _description_
        imgs_key (_type_): _description_
        segs_gt (_type_): _description_
    """
    mask = mergeMultiLabelToOne(segs_gt)
    bin_uint8 = (mask * 255).astype(np.uint8)
    bboxs = mask_find_bboxs(bin_uint8)
    # for b in bboxs:
    try:
        b = bboxs[0]
        x1, y1 = b[0], b[1]
        x2 = b[0] + b[2]
        y2 = b[1] + b[3]
        # print(f'x0:{x0}, y0:{y0}, x1:{x1}, y1:{y1}')
        # start_point, end_point = (x0, y0), (x1, y1)
        ROI_gt = imgs_gt[y1:y2, x1:x2]
        ROI_key = imgs_key[y1:y2, x1:x2]
        return ROI_gt, ROI_key
    except IndexError:
        return imgs_gt, imgs_key

# class Dual_model():
#     def __init__(self, args, acc, mask_type):
#         model_type = "model"
#         data_type = "brain"
#         # mask_type = "radial"
#         w = 0.2 # The weighted parameter
#         bn = False
#         # if acc in [15]:
#         __nclasses, img_size = return_data_ncl_imgsize(args.dataset_name)
#         # else:
#         module=''
#         pre_path = "/home/labuser1/wzw/asl/data/masks"
#         mask = sio.loadmat(pre_path + "/{}_{}_{}_{}.mat".format(mask_type, img_size[0], img_size[1], acc))['Umask']  # 中间低频
#         # mask = sio.loadmat("./DualNet/mask/%s/%s/%s_256_256_%d.mat" % (data_type, mask_type, mask_type, acc))['Umask']
#         path = './model_zoo/tab1/{}/csmri2__{}.pth'.format(args.dataset_name, int(args.rate*100))
#         mask = fftshift(mask, axes=(-2, -1))
        
#         print(path)
        
#         mask_torch = torch.from_numpy(mask).float().cuda()
#         model = MRIReconstruction(mask_torch, w, bn).cuda()
#         if os.path.exists(
#                 path):
#             model.load_state_dict(
#                 torch.load(path))
#             print("Finished load model parameters!")
#         self.model = model
#         self.mask_torch = mask_torch

def plotRoiInImg(img, data_name, namespace=None):
    # roi box
    scale = 2
    roiBox = {}
    # roiBox['ACDC']=(100, 170, 100+50, 170+50)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 240,240 version
    roiBox['ACDC']=(70, 100, 70+30, 100+30)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['BrainTS']=(100, 170, 100+50, 170+50) # # 240,240 version
    roiBox['BrainTS']=(70, 100, 70+30, 100+30)  # # 240,240 version
    # roiBox['OAI']=(100, 170, 100+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    roiBox['OAI']=(190, 170, 190+50, 170+50)
    roiBox['MRB']=(130, 100, 130+50, 100+50) # 右中
    roiBox['MICCAI']=(130, 100, 130+50, 100+50) # 右中
    roiBox['OASI1_MRB']=(130, 100, 130+50, 100+50) # 右中
    roiBox['fastMRI']=(130, 110, 130+50, 110+50) # 右中
    roiBox['Mms']=(70, 100, 70+30, 100+30)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    roiBox['Prostate']=(130, 100, 130+50, 100+50) # 右中
    
    roiBox['MR_ART']=(130, 100, 130+50, 100+50) # 右中
    roiBox['IXI']=(130, 100, 130+50, 100+50) # 右中
    roiBox['MR_ART']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MRB']=(100, 170, 100+50, 170+50) # 中下
    (x1, y1, x2, y2) = roiBox[data_name]

    color1 = (255, 0, 0)

    img2 = img[..., np.newaxis].astype('uint8')
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  # 转换为三通道
    roi = img2[y1:y2, x1:x2]  # numpy 中的 x y 与 cv2和plt的坐标相反。
    roi2 = cv2.resize(roi, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    roi2 = cv2.rectangle(roi2, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)

    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)
    # img2[0:2*(x2-x1), 0:2*(y2-y1)] = roi2 左上
    # img2[-2*(x2-x1)-1:-1, -2*(y2-y1)-1:-1] = roi2 右下
    img2[0:scale*(x2-x1), -scale*(y2-y1)-1:-1] = roi2
    # img2 = cv2.arrowedLine(img2, (140, 80), (170, 40), (255, 0, 0), thickness=2, line_type=cv2.LINE_AA)
    return img2

def plotRoiOutImgTwo(img, channles=1, error_map=False, namespace=None):
    # roi box
            # import cv2
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        # x1,y1 ------
        # |          |
        # |          |
        # |          |
        # --------x2,y2
        # https://learnopencv.com/super-resolution-in-opencv/

    scale = 2
    roiBox = {}
    # roiBox['ACDC']=(100, 170, 100+50, 170+50)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 240,240 version
    # roiBox['ACDC']=(70, 100, 70+30, 100+30)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['BrainTS']=(100, 170, 100+50, 170+50) # # 240,240 version
    # roiBox['BrainTS']=(70, 100, 70+30, 100+30)  # # 240,240 version
    # roiBox['OAI']=(100, 170, 100+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    # roiBox['MRB']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MICCAI']=(130, 100, 130+50, 100+50) # 右中
    
    # fig2.oasis1.2d.0.05
    if namespace:
        roiBox['OASI1_MRB1']=namespace.bbox1 # 右中 长方形
        roiBox['OASI1_MRB2']=namespace.bbox2   # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    else:
        roiBox['OASI1_MRB1']=(128, 100, 128+60, 100+30) # 右中 长方形
        roiBox['OASI1_MRB2']=(95, 170, 95+60, 170+30)   # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['fastMRI']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['Mms']=(70, 100, 70+15, 100+30)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['Prostate']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MRB']=(100, 170, 100+50, 170+50) # 中下

    color1 = 	(255,0,0) # LIGHT GREEN
    color2 = 	(255,240,0)# LIGHT YELLOW

    # color1 = (143,206,0) # LIGHT GREEN
    # color2 = (255,217,102)  # LIGHT YELLOW

    # ROI 1
    (x1, y1, x2, y2) = roiBox['OASI1_MRB1']
    thickness = 1

    
    if channles==1:
        img2 = img[..., np.newaxis].astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  # 转换为三通道
    elif channles==3:
        img2 = img
    roi1 = img2[y1:y2, x1:x2]  # numpy 中的 x y 与 cv2和plt的坐标相反。
    roi11 = cv2.resize(roi1, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    # if not error_map:
        # roi11 = cv2.rectangle(roi11, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=thickness, lineType=cv2.LINE_AA)
    # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)

    # ROI 2
    (x1, y1, x2, y2) = roiBox['OASI1_MRB2']
    roi2 = img2[y1:y2, x1:x2]  # numpy 中的 x y 与 cv2和plt的坐标相反。
    roi22 = cv2.resize(roi2, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    # if not error_map:
        # roi22 = cv2.rectangle(roi22, (0+1, 0+1), (scale*(x2-x1)-2, scale*(y2-y1)-2), color2, thickness=thickness, lineType=cv2.LINE_AA)

    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color2, thickness=2, lineType=cv2.LINE_AA)
    # img2[0:2*(x2-x1), 0:2*(y2-y1)] = roi2 左上
    # img2[-2*(x2-x1)-1:-1, -2*(y2-y1)-1:-1] = roi2 右下

    # img2[0:scale*(x2-x1), -scale*(y2-y1)-1:-1] = roi2  # 在原图中进行放大。
    # img2 = cv2.arrowedLine(img2, (140, 80), (170, 40), (255, 0, 0), thickness=2, line_type=cv2.LINE_AA)
    return img2, roi11, roi22

def plotRoiOutImgACDC(img, channles=1, namespace=None, error_map=False):
    # roi box
            # import cv2
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #    x1,y1 -----(x2, y1)
        #    |          |
        #    |          |
        #    |          |
        # (x1, y2)-------x2,y2
        # https://learnopencv.com/super-resolution-in-opencv/

    scale = 2
    roiBox = {}
    # roiBox['ACDC']=(100, 170, 100+50, 170+100)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 240,240 version
    if namespace:
        roiBox['ACDC'] = namespace.bbox
    else:
        roiBox['ACDC']=(50, 90, 50+120, 90+60)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['BrainTS']=(100, 170, 100+50, 170+50) # # 240,240 version
    # roiBox['BrainTS']=(70, 100, 70+30, 100+30)  # # 240,240 version
    # roiBox['OAI']=(100, 170, 100+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    # roiBox['MRB']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MICCAI']=(130, 100, 130+50, 100+50) # 右中
    
    # fig2.oasis1.2d.0.05
    # roiBox['OASI1_MRB1']=(128, 100, 128+60, 100+30) # 右中 长方形
    # roiBox['OASI1_MRB2']=(95, 170, 95+60, 170+30)   # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['fastMRI']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['Mms']=(70, 100, 70+15, 100+30)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['Prostate']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MRB']=(100, 170, 100+50, 170+50) # 中下

    color1 = 	(255,0,0) # LIGHT GREEN
    color2 = (255,255,0)  # LIGHT YELLOW

    # color1 = (143,206,0) # LIGHT GREEN
    # color2 = (255,217,102)  # LIGHT YELLOW

    # ROI 1
    (x1, y1, x2, y2) = roiBox['ACDC']

    
    if channles==1:
        img2 = img[..., np.newaxis].astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  # 转换为三通道
    elif channles==3:
        img2 = img
    roi1 = img2[y1:y2, x1:x2]  # numpy 中的 x y 与 cv2和plt的坐标相反。
    roi11 = cv2.resize(roi1, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    if not error_map:
        roi11 = cv2.rectangle(roi11, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)
    # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)

    return img2, roi11

def plotRoiOutImgProstate(img, channles=1, error_map=False, namespace=None):
    # roi box
            # import cv2
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #    x1,y1 -----(x2, y1)
        #    |          |
        #    |          |
        #    |          |
        # (x1, y2)-------x2,y2
        # https://learnopencv.com/super-resolution-in-opencv/

    scale = 2
    roiBox = {}
    # roiBox['ACDC']=(100, 170, 100+50, 170+100)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 240,240 version
    if namespace:
        roiBox['Prostate'] = namespace.bbox
    else:
        roiBox['Prostate']=(70, 80, 70+120, 80+60)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['BrainTS']=(100, 170, 100+50, 170+50) # # 240,240 version
    # roiBox['BrainTS']=(70, 100, 70+30, 100+30)  # # 240,240 version
    # roiBox['OAI']=(100, 170, 100+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    # roiBox['OAI']=(190, 170, 190+50, 170+50)
    # roiBox['MRB']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MICCAI']=(130, 100, 130+50, 100+50) # 右中
    
    # fig2.oasis1.2d.0.05
    # roiBox['OASI1_MRB1']=(128, 100, 128+60, 100+30) # 右中 长方形
    # roiBox['OASI1_MRB2']=(95, 170, 95+60, 170+30)   # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['fastMRI']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['Mms']=(70, 100, 70+15, 100+30)  # (x1, y1, x2, y2) x1 -> x2  从左往右；y1 -> y2 从上到下 # 128,241280 version
    # roiBox['Prostate']=(130, 100, 130+50, 100+50) # 右中
    # roiBox['MRB']=(100, 170, 100+50, 170+50) # 中下

    # color1 = (143,206,0) # LIGHT GREEN
    # color2 = (255,217,102)  # LIGHT YELLOW

    color1 = 	(255,0,0) # LIGHT GREEN
    color2 = 	(255,240,0)# LIGHT YELLOW

    # ROI 1
    (x1, y1, x2, y2) = roiBox['Prostate']

    
    if channles==1:
        img2 = img[..., np.newaxis].astype('uint8')
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)  # 转换为三通道
    elif channles==3:
        img2 = img
    roi1 = img2[y1:y2, x1:x2]  # numpy 中的 x y 与 cv2和plt的坐标相反。
    roi11 = cv2.resize(roi1, (scale*(x2-x1), scale*(y2-y1)), interpolation=cv2.INTER_CUBIC)
    if not error_map:
        roi11 = cv2.rectangle(roi11, (0+1, 0+2), (scale*(x2-x1)-1, scale*(y2-y1)-2), color1, thickness=2, lineType=cv2.LINE_AA)
    # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0,0,255), 2)
    img2 = cv2.rectangle(img2, (x1, y1), (x2, y2), color1, thickness=2, lineType=cv2.LINE_AA)

    return img2, roi11

def save_org_for_paper(args, imgs, segs, masks, patient_name, data_name, desired_sparsity):

    error_maps = {}
    # parameters
    error_v = {"ANDI": (0, 100), "Mms": (0, 100), 'OAI':(0, 60), "MRB": (0, 100)}
    vmin, vmax = error_v[data_name]
    # ROI coordinate
    # x1, y1, x2, y2 = 100, 170, 150, 220
    # roiBox = (x1, y1, x2, y2)
    for key in imgs.keys():
        img_temp = imgs[key]
        _img_min = np.min(img_temp)
        _img_max = np.max(img_temp)
        imgs[key] = 255.0 * (img_temp - _img_min) / (_img_max - _img_min)
    for key in imgs.keys():
        if key == 'gt':
            continue
        error_maps[key] = np.abs(imgs['gt'] - imgs[key])    
    
    # save_dir = './' + data_name + '_ckpt/figs/' + patient_name + '/'
    save_dir = args.PREFIX + 'sparsity_{}/{}/' .format(desired_sparsity, patient_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(save_dir + " created successfully.")

    for key in imgs.keys():
        # _img = plotRoiInImg(imgs[key], data_name)
        _img = imgs[key]
        # plt.text(0, 20,'psnr:' + str(round(psnr[key], 2)), color="w"), plt.text(0, 40,'ssim:'+str(round(100*ssim[key], 2)), color="w")
        save_path = save_dir + key + '.png'
        imsave(save_path, _img, dpi=300, vmin=0, vmax=100)
        if key == 'gt':
            continue
        imsave(save_dir + key + '_ermap.png', error_maps[key], dpi=300, vmin=vmin, vmax=vmax)

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir + " created successfully.")

    
def get_mtl_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 
                'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1), 'fastMRI':(0,1), 'Mms':(0,1)}
    vmin, vmax = error_v[data_name]
    # ROI coordinate
    # x1, y1, x2, y2 = 100, 170, 150, 220

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) # ? range 0 ~ 1

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    # plt.subplot(2, 6, 1) # https://morvanzhou.github.io/tutorials/data-manipulation/plt/4-1-subpot1/
    ax = axes_flt[0]
    # imgs['gt'] = cv2.addWeighted(imgs['gt'], 0.5, segs['gt'], 0.5, 50)
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    # im = ax.imshow(segs['gt'], 'gray', alpha=0.15)
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # !!! this
        # num_plot = num_plot + 1
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        # recon img
        # print(ssim[key])
        # plt.subplot(2, 6, num_plot)
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        # err map
        # plt.subplot(2, 6, num_plot+6)
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        # ax.text(0, 20, 'RMSE: '+ str(round(np.sqrt(error_maps[key].mean()), 2)), color="w") # ! text for rmse
        # ax.text(0, 20, 'RMSE: '+ str(round(error_maps[key].mean(), 2)), color="w") # ! text for rmse
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  # 可以取消该区块的坐标轴。

    # https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
    # https://stackoverflow.com/questions/51303380/matplotlib-one-colorbar-for-all-subplots-overlay-on-papersize
    # color bar out of range. The explicit way
    # cb_ax = fig.add_axes([0.91, 0.15, 0.01, 0.69]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    # color bar in subplot[6]
    # plt.colorbar(axes_flt[6], cax=)
    # save_path = './' + data_name +'_ckpt/figs/'+'Figure_'+patient_name+'.png'
    save_path = args.PREFIX + 'figs_{}_{}/{}/rec_{}_{}.png' .format(desired_sparsity, args.maskType, data_name, shotname, NUM_SLICE)
    print(NUM_SLICE)
    # bottom, top = .02, 0.9
    # left, right = 0.1, 0.8
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_error_map(args, imgs, NUM_SLICE, df, data_name, shotname):
    """args, imgs, NUM_SLICE, df, data_name, shotname
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 
                'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1), 'fastMRI':(0,1), 'Mms':(0,1),
                'IXI':{0,1}, 'MR_ART':{0,1}}
    vmin, vmax = error_v[data_name]
    # ROI coordinate
    # x1, y1, x2, y2 = 100, 170, 150, 220

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) # ? range 0 ~ 1

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    # plt.subplot(2, 6, 1) # https://morvanzhou.github.io/tutorials/data-manipulation/plt/4-1-subpot1/
    ax = axes_flt[0]
    # imgs['gt'] = cv2.addWeighted(imgs['gt'], 0.5, segs['gt'], 0.5, 50)
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    # im = ax.imshow(segs['gt'], 'gray', alpha=0.15)
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'Corrupted', 'MARC', 'cycleGAN', 'BSA', 'Ours')
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # !!! this
        # num_plot = num_plot + 1
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        # recon img
        # print(ssim[key])
        # plt.subplot(2, 6, num_plot)
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        # err map
        # plt.subplot(2, 6, num_plot+6)
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        # ax.text(0, 20, 'RMSE: '+ str(round(np.sqrt(error_maps[key].mean()), 2)), color="w") # ! text for rmse
        # ax.text(0, 20, 'RMSE: '+ str(round(error_maps[key].mean(), 2)), color="w") # ! text for rmse
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  # 可以取消该区块的坐标轴。

    # https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
    # https://stackoverflow.com/questions/51303380/matplotlib-one-colorbar-for-all-subplots-overlay-on-papersize
    # color bar out of range. The explicit way
    # cb_ax = fig.add_axes([0.91, 0.15, 0.01, 0.69]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    # color bar in subplot[6]
    # plt.colorbar(axes_flt[6], cax=)
    # save_path = './' + data_name +'_ckpt/figs/'+'Figure_'+patient_name+'.png'
    save_path = args.PREFIX + '/rec_{}_{}.png' .format(shotname, NUM_SLICE)
    print(NUM_SLICE)
    # bottom, top = .02, 0.9
    # left, right = 0.1, 0.8
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_backbone_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1)}
    vmin, vmax = error_v[data_name]
    # ROI coordinate
    # x1, y1, x2, y2 = 100, 170, 150, 220

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) # ? range 0 ~ 1

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    # plt.subplot(2, 6, 1) # https://morvanzhou.github.io/tutorials/data-manipulation/plt/4-1-subpot1/
    ax = axes_flt[0]
    # imgs['gt'] = cv2.addWeighted(imgs['gt'], 0.5, segs['gt'], 0.5, 50)
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    # im = ax.imshow(segs['gt'], 'gray', alpha=0.15)
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'csl_unet', 'asl_unet', 'csl_ista', 'asl_ista', 'csl_mdr')
    # {'csl_unet': None, 'asl_unet': None, 'csl_ista': None, 'asl_ista': None, 'csl_mdr':None, 'asl_mdr':None}
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # !!! this
        # num_plot = num_plot + 1
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        # recon img
        # print(ssim[key])
        # plt.subplot(2, 6, num_plot)
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        # err map
        # plt.subplot(2, 6, num_plot+6)
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        # ax.text(0, 20, 'RMSE: '+ str(round(np.sqrt(error_maps[key].mean()), 2)), color="w") # ! text for rmse
        # ax.text(0, 20, 'RMSE: '+ str(round(error_maps[key].mean(), 2)), color="w") # ! text for rmse
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  # 可以取消该区块的坐标轴。

    # https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
    # https://stackoverflow.com/questions/51303380/matplotlib-one-colorbar-for-all-subplots-overlay-on-papersize
    # color bar out of range. The explicit way
    # cb_ax = fig.add_axes([0.91, 0.15, 0.01, 0.69]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    # color bar in subplot[6]
    # plt.colorbar(axes_flt[6], cax=)
    # save_path = './' + data_name +'_ckpt/figs/'+'Figure_'+patient_name+'.png'
    save_path = args.PREFIX + 'figs_{}_{}/{}/rec_{}_{}.png' .format(desired_sparsity, args.maskType, data_name, shotname, NUM_SLICE)
    print(NUM_SLICE)
    # bottom, top = .02, 0.9
    # left, right = 0.1, 0.8
    # fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_fastmri_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    # NUM_SLICE = NUM_SLICE - 5
    error_maps = {}
    error_v = {"ACDC": (0, 1), "BrainTS": (0, 1), 'OAI':(0, 1), "MRB": (0, 120), 
                'MICCAI':(0, 1), 'OASI1_MRB':(0,1), 'Prostate':(0,1), 'fastMRI':(0,1), 'Mms':(0,1)}
    vmin, vmax = error_v[data_name]
    # ROI coordinate
    # x1, y1, x2, y2 = 100, 170, 150, 220

    for key in imgs.keys():
        if key == 'gt':
            continue

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) /255.0 # ? range 0 ~ 1

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))

    axes_flt = axes.flatten()
    # plt.subplot(2, 6, 1) # https://morvanzhou.github.io/tutorials/data-manipulation/plt/4-1-subpot1/
    ax = axes_flt[0]
    # imgs['gt'] = cv2.addWeighted(imgs['gt'], 0.5, segs['gt'], 0.5, 50)
    im = ax.imshow(plotRoiInImg(img4plot(imgs['gt']), data_name), 'gray'), ax.title.set_text('Ground truth')
    # im = ax.imshow(segs['gt'], 'gray', alpha=0.15)
    ax.set_axis_off()

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # !!! this
        # num_plot = num_plot + 1
        if key == 'gt':
            continue
        num_plot = num_plot + 1
        # recon img
        # print(ssim[key])
        # plt.subplot(2, 6, num_plot)
        ax = axes_flt[num_plot]
        im = ax.imshow(plotRoiInImg(img4plot(imgs[key]), data_name), 'gray'), ax.title.set_text(subplot_name)

        ax.text(0, 20,'PSNR: ' + str(df[key]['PSNR'][NUM_SLICE]), color="w"), ax.text(0, 40,'SSIM: '+str(df[key]['SSIM'][NUM_SLICE]), color="w")
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)
        # err map
        # plt.subplot(2, 6, num_plot+6)
        ax = axes_flt[num_plot+6]
        im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
        # ax.text(0, 20, 'RMSE: '+ str(round(np.sqrt(error_maps[key].mean()), 2)), color="w") # ! text for rmse
        # ax.text(0, 20, 'RMSE: '+ str(round(error_maps[key].mean(), 2)), color="w") # ! text for rmse
        ax.axis('off')
        plt.subplots_adjust(wspace=.02, hspace = .002)

    axes[1, 0].set_axis_off()  # 可以取消该区块的坐标轴。

    # https://jdhao.github.io/2017/06/11/mpl_multiplot_one_colorbar/
    # https://stackoverflow.com/questions/51303380/matplotlib-one-colorbar-for-all-subplots-overlay-on-papersize
    # color bar out of range. The explicit way
    # cb_ax = fig.add_axes([0.91, 0.15, 0.01, 0.69]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    plt.tight_layout()
    cb_ax = fig.add_axes([0.12, 0.08, 0.01, 0.4]) # add an axes, lower left corner in [0.83, 0.2] measured in figure coordinate with axes width 0.01 and height 0.6
    fig.colorbar(im, cax=cb_ax, cmap='jet')
    # color bar in subplot[6]
    # plt.colorbar(axes_flt[6], cax=)
    # save_path = './' + data_name +'_ckpt/figs/'+'Figure_'+patient_name+'.png'
    save_path = args.PREFIX + 'figs_{}_{}/{}/rec_{}_{}.png' .format(desired_sparsity, args.maskType, data_name, shotname, NUM_SLICE)
    print(NUM_SLICE)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)


def get_fastmri_onlyone_error_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 5, 6
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    ax = {}
    fig = plt.figure(figsize=(19, 8))
    # fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(19, 28))

    error_maps = {}
    max_rangs = {}
    error_maps_offset ={}

    vmin, vmax = (0, 1)
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    for key in keyssss:
        max_rangs[key] = np.mean(imgs[key])-np.min(imgs[key])
        # max_rangs[key] = np.max(imgs[key])-np.min(imgs[key])

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        # error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    for key in error_maps.keys():
        error_maps[key] = img4plot(error_maps[key]) /255.0 # ? range 0 ~ 1

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """

        now_col = num_plot

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        img_rec = plotRoiInImg(img4plot(imgs[key]), data_name='fastMRI', namespace=namespace)

        # draw (1,2) row  FULL REC
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        # im = ax.imshow(img_rec, 'gray'), ax.title.set_text(subplot_name)
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        # draw (3,4) row  FULL SEG
        ax = plt.subplot2grid((nrows, ncols), (2, now_col), rowspan=2, colspan=1)
        
        # im = ax.imshow(segmap)
        # ax.set_axis_off()

        ax.set_axis_off()

        if key != 'gt':
        
            im = ax.imshow(error_maps[key], vmin=vmin, vmax=vmax, cmap='jet')
            # ax.set_axis_off()
        if key == 'csmri1':
            cb_ax = fig.add_axes([0.15, 0.21, 0.01, 0.38]) 
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()
            # cbar.ax.set_yticklabels(['< 0', '0.5', '> 1'])  # vertically oriented colorbar

        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_fastmri_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)


def open_morph(imgs):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(imgs, cv2.MORPH_OPEN, kernel)

    for i in range(imgs.shape[0]):
        threshold =np.max(imgs[i]) - np.min(imgs[i]) / 2
        imgs[i][imgs[i] >= threshold] = 1
        imgs[i][imgs[i] < threshold] = 0

    return opening

def tensor2np(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    # img = np.abs(np.squeeze(img))
    img = np.squeeze(img)
    # min = np.min(img) # !
    # max = np.max(img) # !
    # img = 255.0 * ((img - min)/ (max - min)) # !
    return img

def img4plot(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    # img = np.abs(np.squeeze(img)) # ! must
    img = np.squeeze(img)
    min = np.min(img) # !
    max = np.max(img) # !
    img = 255.0 * ((img - min)/ (max - min)) # !
    return img

def normalize(img, max_int=255.0):
    """ normalize image to [0, max_int] according to image intensities [v_min, v_max] """
    v_min, v_max = np.max(img), np.min(img)
    img = (img - v_min)*(max_int)/(v_max - v_min)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img    

def tensor2np4soloseg(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    img = np.squeeze(img)
    return img    

def seg2np(img):
    """ img.shape shoulde be (1, Nx, Ny)
    and normalize img to (256, 256)
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    elif isinstance(img, np.ndarray):
        img = img
    img = np.abs(np.squeeze(img))
    # min = np.min(img)
    # max = np.max(img)
    # img = 255.0 * ((img - min)/ (max - min))
    return img

def save_mask_as_np(args, data_name, traj_type, mode, ckpt, desired_sparsity):
    path = {}
    zero_one_np = {}
    # ckpt = './{}_ckpt/sparsity_{}/{}_{}.pth' .format(data_name, desired_sparsity, mode[0], ckpt[0])
    
    # path['unet_mask'] = './' + data_name +'_ckpt/' + mode[0] +'_%d.pth' % ckpt[0]
    # path['mtl1_mask'] = './' + data_name +'_ckpt/' + mode[1] +'_%d.pth' % ckpt[0]
    # path['unet_mask'] = './{}_ckpt/sparsity_{}/{}_{}.pth' .format(data_name, desired_sparsity, mode[0], ckpt[0])
    # path['mtl1_mask'] = './{}_ckpt/sparsity_{}/{}_{}.pth' .format(data_name, desired_sparsity, mode[1], ckpt[0])

    path['unet_mask'] = args.save_path1
    path['mtl1_mask'] = args.save_path

    img = torch.randn((1, 1, 240, 240)).to(device)
    for key in path.keys():
        if traj_type == "cartesian":
            mask_layer = Mask_Fixed_CartesianLayer(ckpt=path[key], desired_sparsity=desired_sparsity).to(device)
        elif traj_type == "radial":
            pass
        elif traj_type == "random":
            mask_layer = Mask_Fixed_Layer(ckpt=path[key], desired_sparsity=desired_sparsity).to(device)

        (uifft, complex_abs, zero_one, fft, undersample) = mask_layer(img)
        # zero_one_np[key] = tensor2np(zero_one) / 255.0
        zero_one_np[key] = img4plot(zero_one) / 255.0
        zero_one_np[key] = np.fft.fftshift(zero_one_np[key])
        print("undersampling_rate is {}".format(np.mean(zero_one_np[key])))
        # save_path = './' + data_name +'_ckpt/figs/' + key + '.png'
        save_path = args.PREFIX + 'sparsity_{}/{}.png' .format(desired_sparsity, key)
        imsave(save_path, zero_one_np[key], cmap='gray')
    return zero_one_np

def get_mtl_metrics(args, imgs, segs, df, traj_type, shotname, NUM_SLICE, desired_sparsity):
    """
    input: gt = groud truth, zf = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = psnr['zf': 13, 'unet': '30', 'mtl1': '35'], ssim['zf': 45, 'unet': 70, 'mtl1': 90]
    """

    # for key in imgs.keys():
    for key in segs.keys():
        temp_df = init_temp_df(dataset_name=args.dataset_name)
        if key == 'gt':
            continue
        # all picture calculate #
        # max_value = np.array([np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]), np.max(imgs['gt']), np.max(imgs[key])])
        max_value = np.max(imgs['gt'])-np.min(imgs['gt'])
        if key == 'supre':
            temp_df['PSNR'] = None
            temp_df['SSIM'] = None
        else:
            temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate psnr
            temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate ssim
        for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  # calculate dice for multiple segmentation.
            temp_dice = dice_coef(segs[key][i], segs['gt'][i])
            temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  # np.nan
            # temp_df[dfkey] = round(100*temp_dice, 2)
        # all picture calculate #

        # ROI calculate #
        # ROI_gt, ROI_key = get_roi(imgs['gt'], imgs[key], segs['gt'])  # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
        # max_value = np.max(ROI_gt)-np.min(ROI_gt)
        # temp_df['PSNR'] = round(compare_psnr(ROI_gt, ROI_key, data_range=max_value), 2) # calculate psnr
        # temp_df['SSIM'] = round(100*compare_ssim(ROI_gt, ROI_key, data_range=max_value), 2) # calculate ssim
        # for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  # calculate dice for multiple segmentation.
        #     temp_dice = dice_coef(segs[key][i], segs['gt'][i])
        #     temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  # np.nan
        # ROI calculate #

        temp_df['shotname'] = shotname
        temp_df['slice'] = NUM_SLICE
        df[key] = df[key].append(temp_df, ignore_index = True)

def get_metrics_from_out(args, imgs, df, shotname, NUM_SLICE, temp_df, key):
    """
    input: gt = groud truth, zf = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = psnr['zf': 13, 'unet': '30', 'mtl1': '35'], ssim['zf': 45, 'unet': 70, 'mtl1': 90]
    """

    # for key in imgs.keys():
    # for key in imgs.keys():
        # temp_df = init_temp_df(dataset_name=args.dataset_name)
        # if key == 'gt':
        #     continue
        # all picture calculate #
        # max_value = np.array([np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]), np.max(imgs['gt']), np.max(imgs[key])])
        # max_value = np.max(imgs['gt'])-np.min(imgs['gt'])
        # if key == 'supre':
        #     temp_df['PSNR'] = None
        #     temp_df['SSIM'] = None
        # else:
            # temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate psnr
            # temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate ssim

    temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key]), 2) # calculate psnr
    temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key]), 2) # calculate ssim

        # for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  # calculate dice for multiple segmentation.
        #     temp_dice = dice_coef(segs[key][i], segs['gt'][i])
        #     temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  # np.nan
            # temp_df[dfkey] = round(100*temp_dice, 2)
        # all picture calculate #

        # ROI calculate #
        # ROI_gt, ROI_key = get_roi(imgs['gt'], imgs[key], segs['gt'])  # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
        # max_value = np.max(ROI_gt)-np.min(ROI_gt)
        # temp_df['PSNR'] = round(compare_psnr(ROI_gt, ROI_key, data_range=max_value), 2) # calculate psnr
        # temp_df['SSIM'] = round(100*compare_ssim(ROI_gt, ROI_key, data_range=max_value), 2) # calculate ssim
        # for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  # calculate dice for multiple segmentation.
        #     temp_dice = dice_coef(segs[key][i], segs['gt'][i])
        #     temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  # np.nan
        # ROI calculate #
    if key == "Ours":
        temp_df['shotname'] = shotname
        temp_df['slice'] = NUM_SLICE
        df[key] = df[key].append(temp_df, ignore_index = True)

def get_metrics(args, imgs, df, shotname, NUM_SLICE):
    """
    input: gt = groud truth, zf = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = psnr['zf': 13, 'unet': '30', 'mtl1': '35'], ssim['zf': 45, 'unet': 70, 'mtl1': 90]
    """

    # for key in imgs.keys():
    for key in imgs.keys():
        temp_df = init_temp_df(dataset_name=args.dataset_name)
        if key == 'gt':
            continue
        # all picture calculate #
        # max_value = np.array([np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]), np.max(imgs['gt']), np.max(imgs[key])])
        max_value = np.max(imgs['gt'])-np.min(imgs['gt'])
        if key == 'supre':
            temp_df['PSNR'] = None
            temp_df['SSIM'] = None
        else:
            # temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate psnr
            # temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate ssim

            temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key]), 2) # calculate psnr
            temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key]), 2) # calculate ssim

        temp_df['shotname'] = shotname
        temp_df['slice'] = NUM_SLICE
        df[key] = df[key].append(temp_df, ignore_index = True)


def get_ROI_metrics(args, imgs, segs, df, traj_type, shotname, NUM_SLICE, desired_sparsity):
    """
    input: gt = groud truth, zf = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = psnr['zf': 13, 'unet': '30', 'mtl1': '35'], ssim['zf': 45, 'unet': 70, 'mtl1': 90]
    """

    for key in imgs.keys():
        temp_df = init_temp_df(dataset_name=args.dataset_name)
        if key == 'gt':
            continue
        # all picture calculate #
        # max_value = np.array([np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]), np.max(imgs['gt']), np.max(imgs[key])])
        # max_value = np.max(imgs['gt'])-np.min(imgs['gt'])
        # temp_df['PSNR'] = round(compare_psnr(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate psnr
        # temp_df['SSIM'] = round(100*compare_ssim(imgs['gt'], imgs[key], data_range=max_value), 2) # calculate ssim
        # for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  # calculate dice for multiple segmentation.
        #     temp_dice = dice_coef(segs[key][i], segs['gt'][i])
        #     temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  # np.nan
        #     # temp_df[dfkey] = round(100*temp_dice, 2)
        # all picture calculate #

        # ROI calculate #
        ROI_gt, ROI_key = get_roi(imgs['gt'], imgs[key], segs['gt'])  # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
        max_value = np.max(ROI_gt)-np.min(ROI_gt)
        temp_df['PSNR'] = round(compare_psnr(ROI_gt, ROI_key, data_range=max_value), 2) # calculate psnr
        temp_df['SSIM'] = round(100*compare_ssim(ROI_gt, ROI_key, data_range=max_value), 2) # calculate ssim
        for i, dfkey in zip(range(segs['gt'].shape[0]), list(temp_df.keys())[2:]):  # calculate dice for multiple segmentation.
            temp_dice = dice_coef(segs[key][i], segs['gt'][i])
            temp_df[dfkey] = round(100*temp_dice, 2) if temp_dice>0.0001 else np.nan  # np.nan
        # ROI calculate #

        temp_df['shotname'] = shotname
        temp_df['slice'] = NUM_SLICE
        df[key] = df[key].append(temp_df, ignore_index = True)


def get_seg_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # if key == 'dual':
        #     continue
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        # palette[:3*21]=np.array([[0, 0, 0],
        #                             [128, 0, 0],
        #                             [0, 128, 0],
        #                             [128, 128, 0],
        #                             [128, 128, 0],  # WHITE MATTER LESION. [0,0,128]
        #                             [128, 0, 128],
        #                             [0, 128, 128],
        #                             [128, 128, 128],
        #                             [64, 0, 0],
        #                             [192, 0, 0],
        #                             [64, 128, 0],
        #                             [192, 128, 0],
        #                             [64, 0, 128],
        #                             [192, 0, 128],
        #                             [64, 128, 128],
        #                             [192, 128, 128],
        #                             [0, 64, 0],
        #                             [128, 64, 0],
        #                             [0, 192, 0],
        #                             [128, 192, 0],
        #                             [0, 64, 128]], dtype='uint8').flatten()
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], # WM
                                    [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。
        region = imgnp

        ax = axes_flt[num_plot]
        im = ax.imshow(img4plot(imgs[key]), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        ax = axes_flt[num_plot+6]
        
        # first methods for adding seg_results to mri image.
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)

        # https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv
        # reference. PointRend: Image Segmentation as Rendering
        # https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
        # masked_img = np.zeros(imgs[key].shape)
        # for i in range(1, np.max(xx_2ch)):
            # masked_img = np.where(background[i, ..., None]>=0.5, palette_org[i], stacked_img)

        # out = cv2.addWeighted(stacked_img, 0.8, masked_img, 0.2, 0)
        
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        # ax.axis('off')
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)


def get_seg_ablation_map(args, imgs, NUM_SLICE, df, data_name, shotname, keys_list, have_title=True, namespace=None, dpi=300, oneline=False, normal=True):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    keyssss = keys_list
    subplot_name_list = keys_list

    # keyssss = ('gt', 'Corrupted', 'MARC', 'BSA', 'OursPrior','Ours')
    # subplot_name_list = ('Reference', 'Corrupted', 'MARC', 'BSA', 'OursPrior', 'Ours')
    
    if normal:
        img_lists = [normal_gray(imgs[key])  for key, subplot_name in zip(keyssss, subplot_name_list)]
    else:
        img_lists = [imgs[key]  for key, subplot_name in zip(keyssss, subplot_name_list)]

    save_path = args.PREFIX + '/rec_{}_{}.png' .format(shotname, NUM_SLICE)
    num_cols= len(keyssss)

    if oneline:
        plot_oneline(img_lists, 1, num_cols, save_path, subplot_name_list, dpi=dpi, namespace=namespace, have_title=have_title)
    else:
        plot_images_bbox_1and2_is_gray(img_lists, 3, num_cols,save_path, subplot_name_list, dpi=dpi)

def get_seg_oasis1_main_map(args, imgs, NUM_SLICE, df, data_name, shotname, namespace=None, dpi=300, oneline=False):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    keyssss = ('gt', 'Corrupted', 'MARC', 'cycleGAN', 'BSA', 'Ours', 'Supervised')
    subplot_name_list = ('Reference', 'Corrupted', 'MARC', 'cycleGAN', 'BSA', 'Ours', 'Supervised')

    # keyssss = ('gt', 'Corrupted', 'MARC', 'BSA', 'OursPrior','Ours')
    # subplot_name_list = ('Reference', 'Corrupted', 'MARC', 'BSA', 'OursPrior', 'Ours')
    

    img_lists = [normal_gray(imgs[key])  for key, subplot_name in zip(keyssss, subplot_name_list)]
    save_path = args.PREFIX + '/rec_{}_{}.png' .format(shotname, NUM_SLICE)

    num_cols= len(keyssss)

    if oneline:
        plot_oneline(img_lists, 1, num_cols, save_path, subplot_name_list, dpi=dpi)
    else:
        plot_images_bbox_1and2_is_gray(img_lists, 3, num_cols,save_path, subplot_name_list, dpi=dpi)

def get_seg_oasis1_main_map_withdf(args, imgs, NUM_SLICE, df, data_name, shotname, namespace=None, dpi=300, oneline=False):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    keyssss = ('gt', 'Corrupted', 'MARC', 'cycleGAN', 'BSA', 'Ours')
    subplot_name_list = ('Reference', 'Corrupted', 'MARC', 'cycleGAN', 'BSA', 'Ours')

    # keyssss = ('gt', 'Corrupted', 'MARC', 'BSA', 'OursPrior','Ours')
    # subplot_name_list = ('Reference', 'Corrupted', 'MARC', 'BSA', 'OursPrior', 'Ours')
    

    img_lists = [normal_gray(imgs[key])  for key, subplot_name in zip(keyssss, subplot_name_list)]
    save_path = args.PREFIX + '/rec_{}_{}.png' .format(shotname, NUM_SLICE)
    if oneline:
        plot_oneline_with_df(img_lists, 1, 6, save_path, subplot_name_list, args, imgs, df, shotname, NUM_SLICE, dpi=dpi, namespace=namespace)
    else:
        plot_images_bbox_1and2_is_gray_with_df(img_lists, 3,6,save_path, subplot_name_list, dpi=dpi)

def plot_images_bbox_1and2_is_gray(images, rows, cols, save_path, subplot_name_list, dpi=300):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6), dpi=dpi)
    # Define bounding box coordinates for the zoomed-in region
    x1, y1, x2, y2 = 100, 100+40, 190, 190+40
    p1, p2, p3, p4 = 30, 30, 10, 10,


    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols
        if row == 0:  # First row
            img = images[col]
            cmap = 'gray'
            ax.imshow(img, cmap=cmap)
            # Draw bounding box
            if col == 0:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=1, fill=False)
                ax.add_patch(rect)

                # ax.text(0.05, 0.95, 'SNR/EFC/CJV', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')
                ax.text(0.05, 0.95, 'PSNR/SSIM', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')
                # ax.text(0.05, 0.95, 'SSIM', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            elif col > 0:
                psnr = compare_psnr(images[0], images[col])
                ssim = compare_ssim(images[0], images[col])
                psnr_ssim_text = f"{psnr:.2f}/{ssim:.4f}"
                

                # snr = calculate_snr(images[col])
                # efc = calculate_efc(images[col])
                # cjv = calculate_cjv(images[col], images[0])

                # psnr_ssim_text = f"{snr:.2f}/{efc:.4f}/{cjv:.4f}"
                
                # calculate_snr, calculate_efc, calculate_cjv

                # snr = signaltonoise_dB(images[col])
                # lpips_diff = compute_lpips(images[0], images[col])
                # cnr = compute_cnr(images[col], roi=(110, 110, 150, 150))
                

                # psnr_ssim_text = f"{psnr:.2f}/{ssim:.4f}"
                # psnr_ssim_text = f"{ssim:.4f}"
                ax.text(0.05, 0.95, psnr_ssim_text, transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            ax.set_title(subplot_name_list[col])
            

        elif row == 2:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = 'gray'
            vmin, vmax = np.min(img), np.max(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # https://matplotlib.org/3.4.3/gallery/text_labels_and_annotations/arrow_simple_demo.html
            ax.arrow(p1, p2, p3, p4, color='white', width=0.01, head_width=5, length_includes_head=True)

        elif row == 1:  # Last row
            img = images[col]
            cmap = 'seismic'
            # vmin, vmax = np.min(images[0]), np.max(images[0])
            # ax.imshow(img-images[0], cmap=cmap, vmin=-1, vmax=1)
            ref_img = images[0]
            cur_img = img
            diff = (ref_img - cur_img) / (np.max(ref_img) - np.min(ref_img))
            # diff = compute_nrmse(ref_img, cur_img)
            # diff = (diff / np.max(diff) - 0.5 ) *2 # to (-1,1)
            ax.imshow(diff, cmap=cmap, vmin=-1, vmax=1)
            if col == 0:
                ax.text(0.05, 0.95, 'NRMSE', transform=ax.transAxes, fontsize=8, color='red', va='top', ha='left')
            else:
                ax.text(0.05, 0.95, "{:.4f}".format(np.mean(np.abs(diff))), transform=ax.transAxes, fontsize=8, color='red', va='top', ha='left')

        # elif row == 2:  # Last row
        #     img = images[col]
        #     cmap = 'seismic'
        #     # vmin, vmax = np.min(images[0]), np.max(images[0])
        #     # ax.imshow(img-images[0], cmap=cmap, vmin=-1, vmax=1)
        #     ref_img = images[0][y1:y2, x1:x2]
        #     cur_img = images[col][y1:y2, x1:x2]
        #     diff = (ref_img - cur_img) / (np.max(ref_img) - np.min(ref_img))
        #     ax.imshow(diff, cmap=cmap, vmin=-1, vmax=1)
        else:
            ax.axis('off')
            continue

        # elif row == 2:  # Last row (zoomed-in NRMSE)
        #     ref_img = images[0][y1:y2, x1:x2]  # Reference ROI from the first column
        #     cur_img = images[col][y1:y2, x1:x2]  # Current

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(save_path, bbox_inches='tight')

def plot_images_bbox_1and2_is_gray_with_df(images, rows, cols, save_path, subplot_name_list, dpi=300):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6), dpi=dpi)
    # Define bounding box coordinates for the zoomed-in region
    x1, y1, x2, y2 = 100, 100+40, 190, 190+40
    p1, p2, p3, p4 = 30, 30, 10, 10,
    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols
        if row == 0:  # First row
            img = images[col]
            cmap = 'gray'
            ax.imshow(img, cmap=cmap)
            # Draw bounding box
            if col == 0:
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linewidth=1, fill=False)
                ax.add_patch(rect)

                # ax.text(0.05, 0.95, 'SNR/EFC/CJV', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')
                ax.text(0.05, 0.95, 'PSNR/SSIM', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')
                # ax.text(0.05, 0.95, 'SSIM', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            elif col > 0:
                psnr = compare_psnr(images[0], images[col])
                ssim = compare_ssim(images[0], images[col])
                psnr_ssim_text = f"{psnr:.2f}/{ssim:.4f}"
                

                # snr = calculate_snr(images[col])
                # efc = calculate_efc(images[col])
                # cjv = calculate_cjv(images[col], images[0])

                # psnr_ssim_text = f"{snr:.2f}/{efc:.4f}/{cjv:.4f}"
                
                # calculate_snr, calculate_efc, calculate_cjv

                # snr = signaltonoise_dB(images[col])
                # lpips_diff = compute_lpips(images[0], images[col])
                # cnr = compute_cnr(images[col], roi=(110, 110, 150, 150))
                

                # psnr_ssim_text = f"{psnr:.2f}/{ssim:.4f}"
                # psnr_ssim_text = f"{ssim:.4f}"
                ax.text(0.05, 0.95, psnr_ssim_text, transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            ax.set_title(subplot_name_list[col])
            

        elif row == 2:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = 'gray'
            vmin, vmax = np.min(img), np.max(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # https://matplotlib.org/3.4.3/gallery/text_labels_and_annotations/arrow_simple_demo.html
            ax.arrow(p1, p2, p3, p4, color='white', width=0.01, head_width=5, length_includes_head=True)

        elif row == 1:  # Last row
            img = images[col]
            cmap = 'seismic'
            # vmin, vmax = np.min(images[0]), np.max(images[0])
            # ax.imshow(img-images[0], cmap=cmap, vmin=-1, vmax=1)
            ref_img = images[0]
            cur_img = img
            diff = (ref_img - cur_img) / (np.max(ref_img) - np.min(ref_img))
            # diff = compute_nrmse(ref_img, cur_img)
            # diff = (diff / np.max(diff) - 0.5 ) *2 # to (-1,1)
            ax.imshow(diff, cmap=cmap, vmin=-1, vmax=1)
            if col == 0:
                ax.text(0.05, 0.95, 'NRMSE', transform=ax.transAxes, fontsize=8, color='red', va='top', ha='left')
            else:
                ax.text(0.05, 0.95, "{:.4f}".format(np.mean(np.abs(diff))), transform=ax.transAxes, fontsize=8, color='red', va='top', ha='left')

        # elif row == 2:  # Last row
        #     img = images[col]
        #     cmap = 'seismic'
        #     # vmin, vmax = np.min(images[0]), np.max(images[0])
        #     # ax.imshow(img-images[0], cmap=cmap, vmin=-1, vmax=1)
        #     ref_img = images[0][y1:y2, x1:x2]
        #     cur_img = images[col][y1:y2, x1:x2]
        #     diff = (ref_img - cur_img) / (np.max(ref_img) - np.min(ref_img))
        #     ax.imshow(diff, cmap=cmap, vmin=-1, vmax=1)
        else:
            ax.axis('off')
            continue

        # elif row == 2:  # Last row (zoomed-in NRMSE)
        #     ref_img = images[0][y1:y2, x1:x2]  # Reference ROI from the first column
        #     cur_img = images[col][y1:y2, x1:x2]  # Current

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(save_path, bbox_inches='tight')

def plot_oneline_with_df(images, rows, cols, save_path, subplot_name_list, args, imgs, df, shotname, NUM_SLICE, dpi=300, namespace=None):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6), dpi=dpi)
    # Define bounding box coordinates for the zoomed-in region
    if namespace:
        x1, y1, x2, y2 =  namespace.bbox1
    else:
        x1, y1, x2, y2 = 100, 100+40, 190, 190+40

    # x1, y1, x2, y2 = 100, 100+40, 190, 190+40
    p1, p2, p3, p4 = 30, 30, 10, 10

    temp_df = init_temp_df(dataset_name=args.dataset_name)
    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols


        if row == 0:  # First row
            img = deepcopy(images[col])
            cmap = 'gray'
            # ax.imshow(img, cmap=cmap)
            # Draw bounding box
            if col == 0:
                ax.imshow(img, cmap=cmap)
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linestyle='--', linewidth=1, fill=False)
                ax.add_patch(rect)

                ax.text(0.05, 0.95, 'PSNR/SSIM', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            elif col > 0:

                roi = img[y1:y2, x1:x2]
                # Resize the ROI region to fit in the bottom-right corner
                zoom_factor = 1.5
                roi_resized = zoom(roi, zoom_factor)

                # Replace the bottom-right corner of the original image with the resized ROI
                img[-roi_resized.shape[0]:, -roi_resized.shape[1]:] = roi_resized

                ax.imshow(img, cmap=cmap)

                # Add a dashed red rectangle around the zoomed-in region, 2 pixels smaller
                rectangle_x = [img.shape[1] - roi_resized.shape[1] + 2, img.shape[1] - 2, img.shape[1] - 2, img.shape[1] - roi_resized.shape[1] + 2, img.shape[1] - roi_resized.shape[1] + 2]
                rectangle_y = [img.shape[0] - roi_resized.shape[0] + 2, img.shape[0] - roi_resized.shape[0] + 2, img.shape[0] - 2, img.shape[0] - 2, img.shape[0] - roi_resized.shape[0] + 2]
                ax.plot(rectangle_x, rectangle_y, color='red', linestyle='--', linewidth=1)



                psnr = compare_psnr(images[0], images[col])
                ssim = compare_ssim(images[0], images[col])
                
                get_metrics_from_out(args, imgs, df, shotname, NUM_SLICE, temp_df, subplot_name_list[col])  # ? update psnrs and ssims , and get the current metrics - range -1~2000

                psnr_ssim_text = f"{psnr:.2f}/{ssim:.4f}"

                ax.text(0.05, 0.95, psnr_ssim_text, transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            ax.set_title(subplot_name_list[col])

        elif row == 2:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = 'gray'
            vmin, vmax = np.min(img), np.max(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # https://matplotlib.org/3.4.3/gallery/text_labels_and_annotations/arrow_simple_demo.html
            ax.arrow(p1, p2, p3, p4, color='white', width=0.01, head_width=5, length_includes_head=True)

        else:
            ax.axis('off')
            continue

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(save_path, bbox_inches='tight')

def plot_oneline(images, rows, cols, save_path, subplot_name_list, dpi=300, namespace=None, have_title=True):
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6), dpi=dpi)
    # Define bounding box coordinates for the zoomed-in region
    if namespace:
        x1, y1, x2, y2 =  namespace.bbox1
        # x1, y1, x2, y2 = 100, 100+40, 190, 190+40
    else:
        x1, y1, x2, y2 = 100, 100+40, 190, 190+40

    p1, p2, p3, p4 = 30, 30, 10, 10,
    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols
        if row == 0:  # First row
            img = deepcopy(images[col])
            cmap = 'gray'
            # ax.imshow(img, cmap=cmap)
            # Draw bounding box
            if col == 0:
                ax.imshow(img, cmap=cmap)

                if namespace:
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='red', linestyle='--', linewidth=1, fill=False)
                    ax.add_patch(rect)

                ax.text(0.05, 0.95, 'PSNR/SSIM', transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')

            elif col > 0:

                roi = img[y1:y2, x1:x2]
                # Resize the ROI region to fit in the bottom-right corner
                zoom_factor = 1.5
                roi_resized = zoom(roi, zoom_factor)

                # Replace the bottom-right corner of the original image with the resized ROI
                if namespace:
                    img[-roi_resized.shape[0]:, -roi_resized.shape[1]:] = roi_resized

                ax.imshow(img, cmap=cmap)
                if namespace:
                    # Add a dashed red rectangle around the zoomed-in region, 2 pixels smaller
                    rectangle_x = [img.shape[1] - roi_resized.shape[1] + 2, img.shape[1] - 2, img.shape[1] - 2, img.shape[1] - roi_resized.shape[1] + 2, img.shape[1] - roi_resized.shape[1] + 2]
                    rectangle_y = [img.shape[0] - roi_resized.shape[0] + 2, img.shape[0] - roi_resized.shape[0] + 2, img.shape[0] - 2, img.shape[0] - 2, img.shape[0] - roi_resized.shape[0] + 2]
                    ax.plot(rectangle_x, rectangle_y, color='red', linestyle='--', linewidth=1)



                psnr = compare_psnr(images[0], images[col])
                ssim = compare_ssim(images[0], images[col])
                
                psnr_ssim_text = f"{psnr:.2f}/{ssim:.4f}"

                ax.text(0.05, 0.95, psnr_ssim_text, transform=ax.transAxes, fontsize=8, color='white', va='top', ha='left')
            if have_title:
                ax.set_title(subplot_name_list[col])

        elif row == 2:  # Second row (zoomed-in region of the corresponding image in the first row)
            img = images[col][y1:y2, x1:x2]
            cmap = 'gray'
            vmin, vmax = np.min(img), np.max(img)
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            # https://matplotlib.org/3.4.3/gallery/text_labels_and_annotations/arrow_simple_demo.html
            ax.arrow(p1, p2, p3, p4, color='white', width=0.01, head_width=5, length_includes_head=True)

        else:
            ax.axis('off')
            continue

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(save_path, bbox_inches='tight')

def compute_nrmse(image1, image2):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE) between two gray scale images.

    Args:
        image1: a 2D NumPy array representing the first image.
        image2: a 2D NumPy array representing the second image.

    Returns:
        The NRMSE between the two images.

    """
    # Compute the mean squared error between the images
    mse = (image1 - image2) ** 2

    # Compute the root mean squared error between the images
    rmse = np.sqrt(mse)

    # Compute the maximum pixel value of the images
    max_val = np.max([np.max(image1), np.max(image2)])

    # Compute the NRMSE
    nrmse = rmse / max_val

    return nrmse

def normal_gray(still):
    # still = 1.0 * (still - np.min(still)) / (np.max(still) - np.min(still))
    still = still / np.max(still)
    return still

def compute_cnr(image, roi, bg_roi=None):
    """
    Compute the Contrast-to-Noise Ratio (CNR) of an image given a Region of Interest (ROI).

    Args:
        image: a 2D NumPy array representing the image.
        roi: a tuple (x1, y1, x2, y2) representing the bounding box of the ROI.
        bg_roi: a tuple (x1, y1, x2, y2) representing the bounding box of a background region. If None,
                the entire image outside the ROI is used as background.

    Returns:
        The CNR of the image.

    """
    # Extract the ROI and background regions
    roi = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    roi_image = image[roi[1]:roi[3], roi[0]:roi[2]]
    if bg_roi is None:
        bg_roi = (0, 0, image.shape[1], image.shape[0])
    bg_roi = (int(bg_roi[0]), int(bg_roi[1]), int(bg_roi[2]), int(bg_roi[3]))
    bg_image = image[bg_roi[1]:bg_roi[3], bg_roi[0]:bg_roi[2]]

    # Compute the mean and standard deviation of the ROI and background regions
    roi_mean = np.mean(roi_image)
    roi_std = np.std(roi_image)
    bg_mean = np.mean(bg_image)
    bg_std = np.std(bg_image)

    # Compute the CNR
    cnr = abs(roi_mean - bg_mean) / np.sqrt((roi_std ** 2 + bg_std ** 2) / 2)

    return cnr

def compute_lpips(gray_image1, gray_image2):


    # Convert gray scale images to PyTorch tensors
    to_tensor = ToTensor()
    gray_image1 = to_tensor(gray_image1).unsqueeze(0)
    gray_image2 = to_tensor(gray_image2).unsqueeze(0)

    # Load the LPIPS model
    # model = lpips.LPIPS(net='vgg', version='0.1')
    # model.eval()

    # Convert the gray scale images to RGB by repeating the single channel three times
    gray_image1 = gray_image1.repeat(1, 3, 1, 1)
    gray_image2 = gray_image2.repeat(1, 3, 1, 1)

    # Compute the LPIPS distance between the images
    lpips_distance = loss_lpips(gray_image1, gray_image2).mean().item()

    return lpips_distance

def compute_vif(img1, img2):
    # Convert the images to grayscale and float32
    # Compute the mean of the images
    gray1 = img1
    gray2 = img2
    mu1, mu2 = gray1.mean(), gray2.mean()
    # Compute the variances of the images
    var1, var2 = gray1.var(), gray2.var()
    # Compute the covariance of the images
    cov = np.cov(gray1.flatten(), gray2.flatten())[0, 1]
    # Compute the VIF score
    num = (2 * mu1 * mu2 + 0.01) * (2 * cov + 0.03)
    den = (mu1 ** 2 + mu2 ** 2 + 0.01) * (var1 + var2 + 0.03)
    vif = num / den
    return vif

def compute_uqi(img1, img2):
    # Convert the images to grayscale and float32
    gray1 =img1
    gray2 =img2
    # Compute the mean of the images
    mu1, mu2 = gray1.mean(), gray2.mean()
    # Compute the variances of the images
    var1, var2 = gray1.var(), gray2.var()
    # Compute the covariance of the images
    cov = np.cov(gray1.flatten(), gray2.flatten())[0, 1]
    # Compute the luminance similarity
    l = (2 * mu1 * mu2 + 0.01) / (mu1 ** 2 + mu2 ** 2 + 0.01)
    # Compute the contrast similarity
    c = (2 * np.sqrt(var1) * np.sqrt(var2) + 0.01) / (var1 + var2 + 0.01)
    # Compute the structure similarity
    s = (cov + 0.01) / (np.sqrt(var1) * np.sqrt(var2) + 0.01)
    # Compute the UQI
    uqi = l * c * s
    return uqi

def compute_motion_entropy(img):
    # Correcting Bulk In-Plane Motion Artifacts in MRI
    # Using the Point Spread Function
    b_max = np.sqrt(np.sum(np.square(img)))
    E_corrected = -np.sum( (img/b_max) * np.log2(img/b_max+1e-10))
    return E_corrected

def compute_q_motion_quality(motion, ref, corrected):
    E_corrected = compute_motion_entropy(corrected)
    E_motion = compute_motion_entropy(motion)
    E_still = compute_motion_entropy(ref)
    Q_motion = (E_corrected - E_motion) / (E_still -  E_motion)

    return Q_motion


def get_masks_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """

    """
    nrows, ncols = 1, 7
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    ax = {}
    fig = plt.figure(figsize=(9, 4))

    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    # for key, subplot_name in zip(imgs.keys(), subplot_name_list):
    # for key, subplot_name in zip(keyssss, subplot_name_list):

    key = 'csmri1'
    ax = plt.subplot2grid((nrows, ncols), (0,0), rowspan=1, colspan=1)
    
    masks_plot = img4plot(masks[key])
    im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    ax.set_axis_off()

    key = 'csl'
    ax = plt.subplot2grid((nrows, ncols), (0,1), rowspan=1, colspan=3)
    masks_plot = img4plot(masks[key])
    im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    ax.set_axis_off()

    key = 'asl'
    ax = plt.subplot2grid((nrows, ncols), (0,4), rowspan=1, colspan=3)
    masks_plot = img4plot(masks[key])
    im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    ax.set_axis_off()

    save_path = args.PREFIX + 'mask/figs_{}_{}_{}_mask_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_masks_map_onebyone_gridplot(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """

    """
    nrows, ncols = 1, 7
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    # ax = {}
    fig = plt.figure(figsize=(10, 19))

    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    # for key, subplot_name in zip(imgs.keys(), subplot_name_list):
    # for key, subplot_name in zip(keyssss, subplot_name_list):

    key = 'csmri1'
    
    
    masks_plot = np.fft.fftshift(img4plot(masks[key]))

    ax = plt.subplot2grid((nrows, ncols), (0,0), rowspan=1, colspan=1)
    im = ax.imshow(masks_plot, 'gray')
    ax.set_axis_off()
    # plt.xticks([])
    # plt.yticks([])

    key = 'csl'
    for i in range(3):
        
        masks_now = np.fft.fftshift(img4plot(masks[key][i]))

        if i == 0:
            masks_plot = masks_now
            # masks_plot = np.concatenate((masks_now, np.zeros((masks_now.shape[0],4))), axis=-1)
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    ax = plt.subplot2grid((nrows, ncols), (0,1), rowspan=1, colspan=3)
    im = ax.imshow(masks_plot, 'gray')
    ax.set_axis_off()

    key = 'asl'
    # for i in range(3):
    #     ax = plt.subplot2grid((nrows, ncols), (0,4+i), rowspan=1, colspan=1)
    #     masks_plot = img4plot(masks[key][i])
    #     im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    #     ax.set_axis_off()
    for i in range(3):
        masks_now = np.fft.fftshift(img4plot(masks[key][i]))
        if i == 0:
            masks_plot = masks_now
            # masks_plot = np.concatenate((masks_now, np.zeros((masks_now.shape[0],4))), axis=-1)
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    ax = plt.subplot2grid((nrows, ncols), (0,4), rowspan=1, colspan=3)
    im = ax.imshow(masks_plot, 'gray')
    ax.set_axis_off()


    save_path = args.PREFIX + 'mask/figs_{}_{}_{}_mask_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 0.4, left = 0, 
                hspace = 0, wspace = 0)
    # plt.subplots_adjust(wspace = 0.05)

    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gcf().tight_layout()
    plt.savefig(save_path, 
    bbox_inches = 'tight',
        pad_inches = 0
        )

def get_masks_map_onebyone(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """

    """
    nrows, ncols = 1, 7
    # ax = {}
    # fig = plt.figure(figsize=(10, 19))
    # f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'height_ratios': [1, 3, 3]})
    # https://stackoverflow.com/questions/10388462/matplotlib-different-size-subplots
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(10,19), gridspec_kw={'width_ratios': [1, 3, 3]})


    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    # for key, subplot_name in zip(imgs.keys(), subplot_name_list):
    # for key, subplot_name in zip(keyssss, subplot_name_list):

    f_k = np.abs(np.fft.fft2(imgs['gt']))
    f_k = np.log(f_k)

    key = 'csmri1'
    mask_csmri1= img4plot(masks[key])/255.0
    mask_csmri1 = np.rot90(mask_csmri1)
    
    masks_now = np.fft.fftshift(mask_csmri1*f_k)
    masks_now = img4plot(masks_now)
    a0.imshow(masks_now)
    a0.set_axis_off()
    # plt.xticks([])
    # plt.yticks([])

    

    key = 'csl'
    for i in range(3):
        
        masks_now = np.fft.fftshift(img4plot(masks[key][i])/255.0*f_k)
        masks_now = img4plot(masks_now)

        if i == 0:
            masks_plot = masks_now
            # masks_plot = np.concatenate((masks_now, np.zeros((masks_now.shape[0],4))), axis=-1)
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    # a1.imshow(masks_plot, 'gray')
    a1.imshow(masks_plot)
    a1.set_axis_off()

    key = 'asl'
    # for i in range(3):
    #     ax = plt.subplot2grid((nrows, ncols), (0,4+i), rowspan=1, colspan=1)
    #     masks_plot = img4plot(masks[key][i])
    #     im = ax.imshow(np.fft.fftshift(masks_plot), 'gray')
    #     ax.set_axis_off()
    for i in range(3):
        # masks_now = np.fft.fftshift(img4plot(masks[key][i]))
        masks_now = np.fft.fftshift(img4plot(masks[key][i])/255.0*f_k)
        masks_now = img4plot(masks_now)

        if i == 0:
            masks_plot = masks_now
            # masks_plot = np.concatenate((masks_now, np.zeros((masks_now.shape[0],4))), axis=-1)
        else:
            masks_plot = np.concatenate((masks_plot, masks_now), axis=-1)
            
    a2.imshow(masks_plot)
    a2.set_axis_off()


    save_path = args.PREFIX + 'mask/figs_{}_{}_{}_mask_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 0.4, left = 0, 
                hspace = 0, wspace = 0)
    # plt.subplots_adjust(wspace = 0.05)

    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gcf().tight_layout()
    plt.savefig(save_path, 
    bbox_inches = 'tight',
        pad_inches = 0
        )

def get_backbone_seg_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(19, 8))
    # fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'csl_unet', 'asl_unet', 'csl_ista', 'asl_ista', 'csl_mdr', 'asl_mdr')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # if key == 'dual':
        #     continue
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], # WM
                                    [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。
        region = imgnp

        ax = axes_flt[num_plot]
        im = ax.imshow(img4plot(imgs[key]), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        # ax = axes_flt[num_plot+6]
        ax = axes_flt[num_plot+7]
        
        # first methods for adding seg_results to mri image.
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)

        # https://stackoverflow.com/questions/66095686/apply-a-segmentation-mask-through-opencv
        # reference. PointRend: Image Segmentation as Rendering
        # https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend
        # masked_img = np.zeros(imgs[key].shape)
        # for i in range(1, np.max(xx_2ch)):
            # masked_img = np.where(background[i, ..., None]>=0.5, palette_org[i], stacked_img)

        # out = cv2.addWeighted(stacked_img, 0.8, masked_img, 0.2, 0)
        
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        # ax.axis('off')
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_seg_acdc_main_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 5, 6
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    ax = {}
    fig = plt.figure(figsize=(19, 8))
    # fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(19, 28))

    error_maps = {}
    error_maps_offset = {}
    max_rangs = {}

    vmin, vmax = (0, 1)
    # vmin, vmax = (-0.5, 0.5)
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')

    # normalizing a group of images in all ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    group_max = np.max(imgs['gt'])
    group_min = np.min(imgs['gt'])
    for key in keyssss:
        temp_max = np.max(np.abs(imgs[key]))
        temp_min = np.min(np.abs(imgs[key]))
        group_max = max(temp_max, group_max)
        group_min = min(temp_min, group_min)

    max_range = group_max - group_min

    for key in keyssss:
        # max_rangs[key] = np.max(imgs[key])-np.min(imgs[key])
        # max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        gt_normal = (imgs['gt']-group_min) / max_range
        recon_normal = (np.abs(imgs[key])-group_min) / max_range
        error_maps[key] = gt_normal - recon_normal
        # error_maps[key] = (imgs['gt'] - imgs[key]) / max_range # # for complex data, normalizing (0,1)
        # error_maps[key] = error_maps[key]-0.5 # normalizing (-1,1)

    # for key in keyssss:
    #     max_max_range = max(zip(max_rangs.values(), max_rangs.keys()))  # get (2203.0, 'csmri1')
    #     if key == 'asl':
    #         error_maps_offset[key] = max_rangs[key]/max_max_range[0]

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # import cv2
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #    x1,y1 -----(x2, y1)
        #    |          |
        #    |          |
        #    |          |
        # (x1, y2)-------x2,y2
        # https://learnopencv.com/super-resolution-in-opencv/
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.5 # 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],

                                    # [198, 70, 70], # 亮 红蓝绿
                                    # [120, 247, 120], # 
                                    # [21, 21, 149], # WM LESIONS [0, 0, 255]

                                    [85,85,85],
                                    [170,170,170],
                                    [255,255,255],

                                    # [255, 0, 0],
                                    # [0, 255, 0], # WM
                                    # [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        region = img_segmaps_np  # segmap

        now_col = num_plot

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        img_rec, roi1 = plotRoiOutImgACDC(img4plot(imgs[key]), namespace=namespace)

        # draw (1,2) row  FULL REC
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        # im = ax.imshow(img_rec, 'gray'), ax.title.set_text(subplot_name)
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        # draw (3,4) row  FULL SEG
        # ax = plt.subplot2grid((nrows, ncols), (2, now_col), rowspan=2, colspan=1)
        segmap, segroi1 = plotRoiOutImgACDC(region, channles=3, namespace=namespace)
        # im = ax.imshow(segmap)
        # ax.set_axis_off()

        # draw (5) row  ROI REC1
        ax = plt.subplot2grid((nrows, ncols), (2,now_col), rowspan=1, colspan=1)
        im = ax.imshow(roi1, 'gray')
        ax.set_axis_off()

        # draw (6) row  ROI error map1
        ax = plt.subplot2grid((nrows, ncols), (3,now_col), rowspan=1, colspan=1)
        ax.set_axis_off()
        if key != 'gt':
            _img, errorroi1 = plotRoiOutImgACDC(img4plot(error_maps[key]), error_map=True, namespace=namespace)
            errorroi1 = img4plot(cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)) /255.0
            # _img, errorroi1, errorroi2 = plotRoiOutImgTwo(error_maps[key], error_map=True)
            # errorroi1 = cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)
            if key == 'asl':
                # im = ax.imshow(errorroi1 * error_maps_offset[key], vmin=vmin, vmax=vmax, cmap='jet')
                # https://matplotlib.org/stable/tutorials/colors/colormaps.html
                im = ax.imshow(errorroi1, vmin=vmin, vmax=vmax, cmap='bwr')
            else:
                im = ax.imshow(errorroi1, vmin=vmin, vmax=vmax, cmap='bwr')
        if key == 'asl':
            cb_ax = fig.add_axes([0.16, 0.21, 0.004, 0.18]) 
            fig.colorbar(im, cax=cb_ax, cmap='bwr')
            cb_ax.yaxis.tick_left()
            

        # draw (7) row  ROI seg map1
        ax = plt.subplot2grid((nrows, ncols), (4,now_col), rowspan=1, colspan=1)
        im = ax.imshow(segroi1)
        ax.set_axis_off()
        
        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_seg_{}_{}_bwr.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_seg_prostate_main_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity, namespace=None):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 5, 6
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    ax = {}
    fig = plt.figure(figsize=(19, 8))
    # fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(19, 28))

    error_maps = {}
    error_maps_offset = {}
    max_rangs = {}

    vmin, vmax = (0, 1)
    keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    for key in keyssss:
        max_rangs[key] = np.max(imgs[key])-np.min(imgs[key]) # mean better than max.

        max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
        # error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data
        error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data
    for key in keyssss:
        max_max_range = max(zip(max_rangs.values(), max_rangs.keys()))  # get (2203.0, 'csmri1')
        error_maps_offset[key] = max_rangs[key]/max_max_range[0]
        error_maps[key] = img4plot(error_maps[key]) /255.0 # ? range 0 ~ 1

    subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(keyssss, subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # import cv2
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
        #    x1,y1 -----(x2, y1)
        #    |          |
        #    |          |
        #    |          |
        # (x1, y2)-------x2,y2
        # https://learnopencv.com/super-resolution-in-opencv/
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.5 # 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],
                                    [85,85,85],
                                    [170,170,170],
                                    [255,255,255],

                                    # [153,153,153],
                                    # [188,188,188],
                                    # [238,238,238],
                                    # [255, 0, 0],
                                    # [0, 255, 0], # WM
                                    # [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        region = img_segmaps_np  # segmap

        now_col = num_plot

        max_range = np.max(imgs['gt'])-np.min(imgs['gt'])

        img_rec, roi1 = plotRoiOutImgProstate(img4plot(imgs[key]), namespace=namespace)

        # draw (1,2) row  FULL REC
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=2, colspan=1)
        # im = ax.imshow(img_rec, 'gray'), ax.title.set_text(subplot_name)
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        # draw (3,4) row  FULL SEG
        # ax = plt.subplot2grid((nrows, ncols), (2, now_col), rowspan=2, colspan=1)
        segmap, segroi1 = plotRoiOutImgProstate(region, channles=3, namespace=namespace)
        # im = ax.imshow(segmap)
        # ax.set_axis_off()

        # draw (5) row  ROI REC1
        ax = plt.subplot2grid((nrows, ncols), (2,now_col), rowspan=1, colspan=1)
        im = ax.imshow(roi1, 'gray')
        ax.set_axis_off()

        # draw (6) row  ROI error map1
        ax = plt.subplot2grid((nrows, ncols), (3,now_col), rowspan=1, colspan=1)
        ax.set_axis_off()
        if key != 'gt':
            _img, errorroi1 = plotRoiOutImgProstate(img4plot(error_maps[key]), error_map=True, namespace=namespace)
            errorroi1 = img4plot(cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)) /255.0
            # _img, errorroi1, errorroi2 = plotRoiOutImgTwo(error_maps[key], error_map=True)
            # errorroi1 = cv2.cvtColor(errorroi1, cv2.COLOR_RGB2GRAY)
            if key == 'asl':
                im = ax.imshow(errorroi1 * error_maps_offset[key], vmin=vmin, vmax=vmax, cmap='jet')
            else:
                im = ax.imshow(errorroi1, vmin=vmin, vmax=vmax, cmap='jet')
        if key == 'asl':
            cb_ax = fig.add_axes([0.16, 0.21, 0.004, 0.18]) # (left-rigt, up-down, )
            fig.colorbar(im, cax=cb_ax, cmap='jet')
            cb_ax.yaxis.tick_left()

        # draw (7) row  ROI seg map1
        ax = plt.subplot2grid((nrows, ncols), (4,now_col), rowspan=1, colspan=1)
        im = ax.imshow(segroi1)
        ax.set_axis_off()
        
        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_ablation_seg_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))
    fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(19, 8))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'woSegNet', 'woCRec','woSAM', 'woBG', 'woCLS', 'ASL')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # if key == 'dual':
        #     continue
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], # WM
                                    [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。
        region = imgnp

        ax = axes_flt[num_plot]
        im = ax.imshow(img4plot(imgs[key]), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        # ax = axes_flt[num_plot+6]
        ax = axes_flt[num_plot+7]
        
        # first methods for adding seg_results to mri image.
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)
       
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        # ax.axis('off')
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_ablation_seg_one_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 2, 7
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    ax = {}
    fig = plt.figure(figsize=(16, 4.5))
    # fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(19, 28))

    # error_maps = {}
    # max_rangs = {}
    # error_maps_offset ={}

    # vmin, vmax = (0, 1)
    # keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    # for key in keyssss:
    #     max_rangs[key] = np.mean(imgs[key])-np.min(imgs[key])
    #     # max_rangs[key] = np.max(imgs[key])-np.min(imgs[key])

    #     max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
    #     # error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data
    #     error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    # for key in error_maps.keys():
    #     error_maps[key] = img4plot(error_maps[key]) /255.0 # ? range 0 ~ 1
    subplot_name_list = ('Ground truth', 'woSegNet', 'woCRec','woSAM', 'woBG', 'woCLS', 'ASL')
    # subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.3 # 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],
                                    # [153,153,153], # (白，灰，黑)
                                    # [188,188,188],
                                    # [238,238,238],

                                    # [255, 0, 0], # （）红 绿 蓝
                                    # [0, 255, 0], # WM
                                    # [0, 0, 255], # WM LESIONS [0, 0, 255]

                                    # [125, 0, 0], # （）红 绿 蓝
                                    # [0, 125, 0], # WM
                                    # [0, 0, 125], # WM LESIONS [0, 0, 255]

                                    [198, 70, 70], # 亮 红蓝绿
                                    [120, 247, 120], # 
                                    [21, 21, 149], # WM LESIONS [0, 0, 255]

                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。
        region = img_segmaps_np  # segmap

        now_col = num_plot

        # draw (1,2) row  FULL REC

        img_rec = img4plot(imgs[key])
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=1, colspan=1)
        # im = ax.imshow(img_rec, 'gray'), ax.title.set_text(subplot_name)
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        # draw (3,4) row  FULL SEG
        ax = plt.subplot2grid((nrows, ncols), (1, now_col), rowspan=1, colspan=1)
        # stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        # seg_map = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        segmap = region
        im = ax.imshow(segmap)
        ax.set_axis_off()

        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_ablatiom_module_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)

def get_ablation_seg_one_map(args, imgs, segs, masks, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """
    nrows, ncols = 2, 7
    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig = plt.figure(figsize=(19, 28), dpi=300)
    ax = {}
    fig = plt.figure(figsize=(16, 4.5))
    # fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(19, 28))

    # error_maps = {}
    # max_rangs = {}
    # error_maps_offset ={}

    # vmin, vmax = (0, 1)
    # keyssss = ('gt', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    # for key in keyssss:
    #     max_rangs[key] = np.mean(imgs[key])-np.min(imgs[key])
    #     # max_rangs[key] = np.max(imgs[key])-np.min(imgs[key])

    #     max_range = max(np.max(imgs['gt'])-np.min(imgs['gt']), np.max(imgs[key])-np.min(imgs[key]))
    #     # error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data
    #     error_maps[key] = np.abs(((imgs['gt'] - imgs[key]) / max_range)) # # for complex data

    # for key in error_maps.keys():
    #     error_maps[key] = img4plot(error_maps[key]) /255.0 # ? range 0 ~ 1
    subplot_name_list = ('Ground truth', 'csl_unet', 'asl_unet', 'csl_ista', 'asl_ista', 'csl_mdr', 'asl_mdr')
    # subplot_name_list = ('Ground truth', 'csmri1', 'csmtl', 'csmri2', 'csl', 'asl')
    num_plot = 0

    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        _seg = segs[key]
        imgsize = _seg.shape
        seg_threshold = 0.3 # 
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * seg_threshold # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img_segmaps = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))

        palette_org = np.array([[0, 0, 0],

                                    [198, 70, 70], # 亮 红蓝绿
                                    [120, 247, 120], # 
                                    [21, 21, 149], # WM LESIONS [0, 0, 255]

                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img_segmaps.putpalette(palette)
        img_segmaps = img_segmaps.convert('RGB')
        img_segmaps_np = np.array(img_segmaps)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。
        region = img_segmaps_np  # segmap

        now_col = num_plot

        # draw (1,2) row  FULL REC

        img_rec = img4plot(imgs[key])
        ax = plt.subplot2grid((nrows, ncols), (0,now_col), rowspan=1, colspan=1)
        # im = ax.imshow(img_rec, 'gray'), ax.title.set_text(subplot_name)
        im = ax.imshow(img_rec, 'gray')
        ax.set_axis_off()

        # draw (3,4) row  FULL SEG
        ax = plt.subplot2grid((nrows, ncols), (1, now_col), rowspan=1, colspan=1)
        # stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        # seg_map = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        segmap = region
        im = ax.imshow(segmap)
        ax.set_axis_off()

        num_plot = num_plot + 1
    save_path = args.PREFIX + 'figs_{}_{}_{}_ablation_backbone_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)

    # plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(save_path, bbox_inches = 'tight',
        pad_inches = 0)


def get_ablation_step_seg_map(args, imgs, segs, masks, zero_recons, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))
    ncols = 4
    fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(16, 16))
    axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'Step1', 'Step2', 'Step3')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # if key == 'dual':
        #     continue
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],
                                    [255, 0, 0],
                                    [0, 255, 0], # WM
                                    [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。

        # Mask
        ax = axes_flt[num_plot+ncols*0]
        if key == 'gt':
            pass
        else:
            shift_mask = np.fft.ifftshift(masks[key])
            # if args.maskType == '1D':
                # shift_mask = shift_mask[0:20, ...]
            im = ax.imshow(img4plot(shift_mask), 'gray'), ax.title.set_text(subplot_name)
        ax.set_axis_off()

        # ZF
        ax = axes_flt[num_plot+ncols*1]
        if key == 'gt':
            pass
        else:
            im = ax.imshow(img4plot(zero_recons[key]), 'gray')
        ax.set_axis_off()
        
        # Rec
        ax = axes_flt[num_plot+ncols*2]
        region = imgnp

        im = ax.imshow(img4plot(imgs[key]), 'gray')
        ax.set_axis_off()

        # Seg
        # ax = axes_flt[num_plot+6]
        ax = axes_flt[num_plot+ncols*3]
        
        # first methods for adding seg_results to mri image.
        stacked_img = cv2.cvtColor(img4plot(imgs[key]).astype('uint8'), cv2.COLOR_GRAY2RGB)
        picture = cv2.addWeighted(stacked_img, 1.0, region.astype('uint8'), 0.5, 0.0)
        im = ax.imshow(picture)
       
        ax.set_axis_off()
        plt.subplots_adjust(wspace=.02, hspace = .002)

        num_plot = num_plot + 1
        # ax.axis('off')
    plt.tight_layout()
    save_path = args.PREFIX + 'figs_{}_{}/{}/seg_{}_{}.png' .format(desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
    plt.margins(0, 0)
    plt.savefig(save_path)

def get_fig1_map(args, imgs, segs, masks, zero_recons,masked_kspaces, NUM_SLICE, df, data_name, shotname, desired_sparsity):
    """
    input is dict: gt = groud truth, zf_unet = zero-filling, unet = loupe + recon_unet, img_mtl1 = loupe + recon_unet + seg_unet
    output = figure [gt, (gt - zerofilling), (gt - unet), (gt - mtl1)]
    TODO: Reduce left and right margins in matplotlib plot
    """

    # plt.figure(figsize=(19, 10))  # 画面大小其实是 1900 * 1000的,（所以，不要太输入太的数字）
    # fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(19, 8))
    # ncols = 4
    # fig, axes = plt.subplots(nrows=4, ncols=ncols, figsize=(16, 16))
    # axes_flt = axes.flatten()    

    subplot_name_list = ('Ground truth', 'Step1', 'Step2', 'Step3')
    num_plot = 0
    
    for key, subplot_name in zip(imgs.keys(), subplot_name_list):
        """ plot img from 1 ~ 5:  1:baseline,  2:liu,  3:dual,  4:loupe,  5:ours
        """
        # if key == 'dual':
        #     continue
        _seg = segs[key]
        imgsize = _seg.shape
        background = np.ones((imgsize[0]+1, imgsize[1], imgsize[2])) * 0.5 # 在argmax中，比0.5大就是该标签，比0.5小，就是背景。
        background[1:imgsize[1]+1] = _seg[0:imgsize[1]]
        xx = np.argmax(background, axis=0)
        xx_2ch = xx.astype(np.uint8)
        xx = np.squeeze(xx_2ch)
        img = Image.fromarray(xx, mode="P")
        palette=[]
        for i in range(256):
            palette.extend((i,i,i))
        palette_org = np.array([[0, 0, 0],

                                    [85,85,85],
                                    [255,255,255],
                                    [170,170,170],
                                    
                                    # [255, 0, 0],
                                    # [0, 255, 0], # WM
                                    # [0, 0, 255], # WM LESIONS [0, 0, 255]
                                    [128, 128, 0],  
                                    [128, 0, 128],
                                    [0, 128, 128],
                                    [128, 128, 128],
                                    [64, 0, 0],
                                    [192, 0, 0],
                                    [64, 128, 0],
                                    [192, 128, 0],
                                    [64, 0, 128],
                                    [192, 0, 128],
                                    [64, 128, 128],
                                    [192, 128, 128],
                                    [0, 64, 0],
                                    [128, 64, 0],
                                    [0, 192, 0],
                                    [128, 192, 0],
                                    [0, 64, 128]], dtype='uint8')
        palette[:3*21]=palette_org.flatten()

        img.putpalette(palette)
        img = img.convert('RGB')
        imgnp = np.array(img)
        delta = 10
        # rregion = imgnp[int(imgnp.shape[0]/2-imgnp.shape[0]/3+10):int(imgnp.shape[0]/2+imgnp.shape[0]/3+10),
        #                int(imgnp.shape[1]/2-imgnp.shape[1]/3):int(imgnp.shape[1]/2+imgnp.shape[1]/3)]
        # 放大可视范围。
        dpi = 200
        # Mask
        if key == 'gt':
            pass
        else:
            shift_mask = np.fft.ifftshift(masks[key])

            save_path = args.PREFIX + 'mask_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
            imsave(save_path, img4plot(shift_mask), cmap='gray', dpi=dpi)


        # ZF
        if key == 'gt':
            pass
        else:
            save_path = args.PREFIX + 'zf_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
            imsave(save_path, img4plot(zero_recons[key]), cmap='gray', dpi=dpi)

        # u_k
        if key == 'gt':
            pass
        else:
            # u_img = zero_recons[key]
            # u_k = np.abs(np.fft.fft2(u_img))
            # u_k = np.log(np.fft.ifftshift(u_k))
            # u_k = shift_mask * u_k
            
            # u_k = np.sqrt(masked_kspaces[key], ord='fro', axis=0)

            # u_k = np.log(np.fft.ifftshift(u_k))
            epsilon = 1e-8
            u_k = np.log(np.fft.ifftshift(masked_kspaces[key] + epsilon))
            

            save_path = args.PREFIX + 'uk_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
            imsave(save_path, img4plot(u_k), dpi=dpi)

        # Rec
        save_path = args.PREFIX + 'rec_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
        imsave(save_path, img4plot(imgs[key]), cmap='gray', dpi=300)

        # Seg
        save_path = args.PREFIX + 'seg_{}_{}_{}_{}_seg_{}_{}.png' .format(key, desired_sparsity, args.maskType, data_name,shotname, NUM_SLICE)
        imsave(save_path, imgnp, dpi=dpi)

        num_plot = num_plot + 1




def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(dir + " created successfully.")

def newdf2excel(file_name, df):
    with pd.ExcelWriter(file_name) as writer:
        for key in df.keys():
            df[key].to_excel(writer, sheet_name=key) 

def dataframe_template(dataset_name):
    df_general = {"PSNR": [], "SSIM": []}
    df_specific_dataset = {}
    
    if dataset_name == 'MRB':
        df_specific_dataset = {"Cortical GM": [], "Basal ganglia": [], "WM": [], "WM lesions": [], "CSF": [], "Ventricles": [], "Cerebellum": [], "Brainstem": []}  # !
    elif dataset_name == 'OAI':
        df_specific_dataset = {"Femoral Cart.": [], "Medial Tibial Cart.": [], "Lateral Tibial Cart.": [], "Patellar Cart.": [], "Lateral Meniscus": [], "Medial Meniscus": []}
    elif dataset_name == 'ACDC':
        df_specific_dataset = {"LV": [], "RV": [], "MYO": []}
    elif dataset_name == 'BrainTS':
        #  Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2),
        #  and the necrotic and non-enhancing tumor core (NCR/NET — label 1),
        df_specific_dataset = {"NCR/NET": [], "ED": [], "ET": []}
    elif dataset_name == "MICCAI":
        df_specific_dataset = {"1": [], "2": [], "3": [], "4": [], "5": [], "6": [], "7":[], "8": [], "9": [], "10": [],
                                "11":[], "12":[], "13":[], "14": [], "15":[], "16": [], "17": [], "18": [], "19": [], "20": [],
                                "21": [], "21": [], "22":[], "23": [], "24":[], "25":[], "26":[], "27":[], "28":[], "29":[], "30":[],
                                "31": [], "32":[], "33":[]}
    elif dataset_name == 'OASI1_MRB':
        df_specific_dataset = {"GM": [], "WM": [], "CSF": []}

    elif dataset_name == 'Prostate':
        df_specific_dataset = {"Seg1": [], "Seg2": []}
    df_last = {"shotname": [], "slice": []} # FID, AVD, HF-95
    df = {**df_general, **df_specific_dataset, **df_last}
    return df

def init_temp_df_drop():
    temp_df = {"PSNR": None, "SSIM": None, "Cortical GM": None, "Basal ganglia": None, "WM": None, "WM lesions": 0,  # merge WM and WM lesions
    "CSF": None,
    "Ventricles": None,
    "Cerebellum": None,
    "Brainstem": None
    }
    return temp_df

def init_temp_df(dataset_name):
    df_general = {"PSNR": None, "SSIM": None}
    df_specific_dataset = {}
    if dataset_name == 'MRB':
        df_specific_dataset = {"Cortical GM": None, "Basal ganglia": None, "WM": None, "WM lesions": None, "CSF": None, "Ventricles": None, "Cerebellum": None, "Brainstem": None}  # !
    elif dataset_name == 'OAI':
        df_specific_dataset = {"Femoral Cart.": None, "Medial Tibial Cart.": None, "Lateral Tibial Cart.": None, "Patellar Cart.": None, "Lateral Meniscus": None, "Medial Meniscus": None}
    elif dataset_name == 'ACDC':
        df_specific_dataset = {"LV": None, "RV": None, "MYO":None}
    elif dataset_name == 'Mms':
        df_specific_dataset = {"LV": None, "RV": None, "MYO":None}
    elif dataset_name == 'BrainTS':
        #  Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2),
        #  and the necrotic and non-enhancing tumor core (NCR/NET — label 1),
        df_specific_dataset = {"NCR/NET": None, "ED": None, "ET": None}
    elif dataset_name == "MICCAI":
        df_specific_dataset = {"1": None, "2": None, "3": None, "4": None, "5": None, "6": None, "7":None, "8": None, "9": None, "10": None,
                                "11":None, "12":None, "13":None, "14": None, "15":None, "16": None, "17": None, "18": None, "19": None, "20": None,
                                "21": None, "21": None, "22":None, "23": None, "24":None, "25":None, "26":None, "27":None, "28":None, "29":None, "30":None,
                                "31": None, "32":None, "33":None}
    elif dataset_name == 'OASI1_MRB':
        df_specific_dataset = {"GM": None, "WM": None, "CSF": None}
    elif dataset_name == 'fastMRI':
        df_specific_dataset = {"GM": None, "WM": None, "CSF": None}
    elif dataset_name == 'Prostate':
        df_specific_dataset = {"Seg1": None, "Seg2": None}

    df_last = {"shotname": None, "slice": None} # FID, AVD, HF-95
    df = {**df_general, **df_specific_dataset, **df_last}
    return df

def get_rice_noise(img, snr=10, mu=0.0, sigma=1):
    # https://blog.csdn.net/howlclat/article/details/107216722 正确地为图像添加高斯噪声
    level = snr * torch.max(img) / 3000
    # sigma = 1.0
    # sigma=std (in 2022-1519-review-tmi, sigma=0.5, 1.0, 1.5), var, mean=0
    # real_part = img[0:1, 0:1, ...]
    # img_part =  img[0:1, 1:2, ...]
    size = img.shape
    x = level * torch.randn(size).to(device, dtype=torch.float) * sigma
    y = level * torch.randn(size).to(device, dtype=torch.float) * sigma
    x = x + img

    return torch.sqrt(x**2 + y**2)


def my_normalized(img, range=1):
    min = np.min(img) # !
    max = np.max(img) # !
    img = range * ((img - min)/ (max - min)) # !
    return img

def otsu_threshold(image=None, hist=None):
    """ Runs the Otsu threshold algorithm.

    Reference:
    Otsu, Nobuyuki. "A threshold selection method from gray-level
    histograms." IEEE transactions on systems, man, and cybernetics
    9.1 (1979): 62-66.

    @param image: The input image
    @type image: ndarray
    @param hist: The input image histogram
    @type hist: ndarray

    @return: The Otsu threshold
    @rtype int
    https://github.com/manuelaguadomtz/pythreshold/blob/master/pythreshold/global_th/otsu.py
    """
    if image is None and hist is None:
        raise ValueError('You must pass as a parameter either'
                         'the input image or its histogram')

    # Calculating histogram
    if not hist:
        hist = np.histogram(image, bins=range(256))[0].astype(np.float)

    cdf_backg = np.cumsum(np.arange(len(hist)) * hist)
    w_backg = np.cumsum(hist)  # The number of background pixels
    w_backg[w_backg == 0] = 1  # To avoid divisions by zero
    m_backg = cdf_backg / w_backg  # The means

    cdf_foreg = cdf_backg[-1] - cdf_backg
    w_foreg = w_backg[-1] - w_backg  # The number of foreground pixels
    w_foreg[w_foreg == 0] = 1  # To avoid divisions by zero
    m_foreg = cdf_foreg / w_foreg  # The means

    var_between_classes = w_backg * w_foreg * (m_backg - m_foreg) ** 2

    return np.argmax(var_between_classes)

def compared_PE_line(gt_img, rec_img, W=240, random_mask=False):


    gt_fft = np.fft.fftn(gt_img, axes=(0,1)) # 
    gt_fft = np.fft.fftshift(gt_fft, axes=(0,1)) # 
    rec_fft = np.fft.fftn(rec_img, axes=(0,1)) # 
    rec_fft = np.fft.fftshift(rec_fft, axes=(0,1)) # 

    gt_fft_abs = np.abs(gt_fft)
    rec_fft_abs = np.abs(rec_fft)
    h, w = gt_fft_abs.shape

    v_range = 255
    gt_fft = my_normalized(gt_fft_abs, v_range)
    rec_fft = my_normalized(rec_fft_abs, v_range)

    pe_psnr = [compare_psnr(gt_fft_abs[i],rec_fft_abs[i], data_range=v_range) for i in range(h)]

    pe_psnr = np.array(pe_psnr)
    pe_psnr = pe_psnr[:, np.newaxis]

    otsu_thresh = otsu_threshold(pe_psnr)

    # pe_psnr[pe_psnr > otsu_thresh] = 1
    # pe_psnr[pe_psnr <= otsu_thresh] = 0

    idx_mask = np.where(pe_psnr > otsu_thresh, 1, 0)

    # mask_from_pe_line = 1-pe_psnr

    mask_from_pe_line = 1-idx_mask

    mask_from_pe_line = np.repeat(mask_from_pe_line, W, axis=1)


    if random_mask:
        path = "./data/masks/motion33mask/bsa/random/cartesian_240_240_33_1.mat"
        trajectory = sio.loadmat(path)['Umask']
        mask_from_pe_line = mask_from_pe_line*trajectory


    sub_k = mask_from_pe_line * rec_fft

    return mask_from_pe_line, sub_k



if __name__ == "__main__":
    pass