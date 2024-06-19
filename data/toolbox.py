import os
import numpy as np
import cv2
import torch
import glob
import h5py
import sys
sys.path.append('.') 

def crop_arr(arr, size):
    """crop img_arr in dataloader. before transform.; CENTERCROP
        arr = (h, w), size is target size.
    """
    h, w = arr.shape[0], arr.shape[1]
    th, tw = size[0], size[1]
    crop_img = arr[int(h/2)-int(th/2):int(h/2)+int(th/2), int(w/2)-int(tw/2):int(w/2)+int(tw/2)]
    return crop_img

def resize_vol_img(arr, size):
    """crop img_arr in dataloader. before transform.; CENTERCROP
        arr = (slice, h, w), size is target size.
    """
    s, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    new_arr = np.zeros((s, size, size))
    for i in range(arr.shape[0]):
        new_arr[i] = cv2.resize(arr[i], (size, size), interpolation=cv2.INTER_LINEAR)
    return new_arr

def resize_vol_seg(arr, size):
    """resize img_arr in dataloader. before transform.; CENTERCROP
        arr = (slice, channel,h, w), size is target size.
    """
    s, c, h, w = arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3]
    new_arr = np.zeros((s, c, size, size))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            new_arr[i][j] = cv2.resize(arr[i][j], (size, size), interpolation=cv2.INTER_LINEAR)
    return new_arr

def get_filePath_fileName_fileExt(fileUrl):
    filepath, tmpfilename = os.path.split(fileUrl)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension

def get_mrb_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    if 'MRBrainS13DataNii' in filepath: 
        realshotname = filepath.split("/")[-1]
    elif '18training_corrected' in filepath:
        realshotname = filepath.split("/")[-2]
    
    return realshotname

def get_acdc_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    
    return realshotname

def get_braints_shotname(fileUrl):
    filepath, _shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    realshotname = filepath.split("/")[-1]

    print(realshotname)
    return realshotname

def get_oai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    print(shotname)
    return shotname

def get_miccai_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)
    

    
    return shotname

def get_h5py_shotname(fileUrl):
    filepath, shotname, _extension = get_filePath_fileName_fileExt(fileUrl)

    # print(shotname)
    return shotname

def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def torch0to1(img):
    B, C, H, W = img.shape
    for b in range(B):
        img_min = torch.min(img[b, :, :,:])
        img_max = torch.max(img[b, :, :,:])
        img[b, :, :,:] = 1.0 * (img[b, :, :,:] - img_min) / (img_max - img_min)
    return img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


class OASI1_MRB(object):

    @staticmethod
    def get_oasi1mrb_edge_h5py():
        raw_path = r'./data/datasets/brain/OASI1_MRB/'
        
        
        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        
        test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')

        return train_img_path, test_img_path
    
    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            img_ = data['img'][slice]
            seg_ = data['seg'][slice]
            edge = data['edge'][slice]
            img_ = img_.astype(np.double)
            seg_ = seg_.astype(np.double)
            edge = edge.astype(np.double)
            return img_, seg_, edge

class IXI_T1_motion_old(object):
    @staticmethod
    def get_ixi_t1_motion_h5py():
        raw_path = r'./data/datasets/IXI_T1_motion/'
        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')
        return train_img_path, test_img_path
    
    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            still = data['still'][slice]
            with_motion = data['with_motion'][slice]

            still = still.astype(np.double)
            with_motion = with_motion.astype(np.double)
            return still, with_motion

class IXI_T1_motion(object):
    @staticmethod
    def get_ixi_t1_motion_h5py(subdir=None):
        raw_path = r'./data/datasets/IXI_T1_motion/'
        if subdir:
            raw_path = r'./data/datasets/IXI_T1_motion/' + subdir
        train_img_path = glob.glob(raw_path+'/training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'/testing-h5py/*.h5')
        return train_img_path, test_img_path
    
    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            still = data['still'][slice]
            with_motion = data['with_motion'][slice]

            still = still.astype(np.double)
            with_motion = with_motion.astype(np.double)
            return still, with_motion
    @staticmethod
    def getsliceandframefromh5py(fname, slice):
        frame, slice_c = slice
        with h5py.File(fname, 'r') as data:
            still = data['still'][frame][slice_c]
            with_motion = data['with_motion'][frame][slice_c]

            still = still.astype(np.double)
            with_motion = with_motion.astype(np.double)
            return still, with_motion

    @staticmethod
    def getslicefromh5py_coronal(fname, slice):
        with h5py.File(fname, 'r') as data:
            still = data['still'][:,slice,:]
            with_motion = data['with_motion'][:,slice,:]

            still = still.astype(np.double)
            with_motion = with_motion.astype(np.double)
            return still, with_motion
        
    @staticmethod
    def getslicefromh5py_sagittal(fname, slice):
        with h5py.File(fname, 'r') as data:
            still = data['still'][:,:,slice]
            with_motion = data['with_motion'][:,:,slice]

            still = still.astype(np.double)
            with_motion = with_motion.astype(np.double)
            return still, with_motion

    @staticmethod
    def get_respiratory_motion_h5py(subdir=None):
        raw_path = r'./data/datasets/IXI_T1_motion/respiratory/'
        if subdir:
            raw_path = r'./data/datasets/IXI_T1_motion/respiratory/' + subdir
        train_img_path = glob.glob(raw_path+'/training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'/testing-h5py/*.h5')
        # test_img_path = glob.glob(raw_path+'/testing-h5py/TCGA-G3-AAV7_pre.h5')
        return train_img_path, test_img_path
    @staticmethod
    def get_cardiac_motion_h5py(subdir=None):
        raw_path = r'./data/datasets/IXI_T1_motion/cardiac/'
        if subdir:
            raw_path = r'./data/datasets/IXI_T1_motion/cardiac/' + subdir
        train_img_path = glob.glob(raw_path+'/training-h5py/*.h5')

        test_img_path = glob.glob(raw_path+'/testing-h5py/*.h5') # 💥

        # test_img_path = glob.glob(raw_path+'/testing-h5py/TCGA-G3-AAV7_pre.h5')
        # test_img_path = glob.glob(raw_path+'/testing-h5py/patient002_4d.nii.h5')

        # test_img_path = glob.glob(raw_path+'/testing-h5py/patient012_4d.nii.h5') # this💎

        # test_img_path = glob.glob(raw_path+'/testing-h5py/patient005_4d.nii.h5')
        return train_img_path, test_img_path



class MR_ART(object):
    @staticmethod
    def get_mr_art_h5py():
        raw_path = r'./data/datasets/MR-ART/'
        train_img_path = glob.glob(raw_path+'training-h5py/*.h5')
        test_img_path = glob.glob(raw_path+'testing-h5py/*.h5')
        return train_img_path, test_img_path
        
    @staticmethod
    def getslicefromh5py(fname, slice):
        with h5py.File(fname, 'r') as data:
            headmotion1 = data['headmotion1'][slice]
            headmotion2 = data['headmotion2'][slice]
            standard = data['standard'][slice]

            headmotion1 = headmotion1.astype(np.double)
            headmotion2 = headmotion2.astype(np.double)
            standard = standard.astype(np.double)
            return headmotion1, headmotion2, standard