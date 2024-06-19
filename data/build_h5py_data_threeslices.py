import sys
import numpy as np
import warnings
import random
import h5py

import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

sys.path.append('.') 
from utils.visdom_visualizer import VisdomLinePlotter
from data.toolbox import get_h5py_shotname
from data.toolbox import OASI1_MRB, IXI_T1_motion, MR_ART
from tqdm import tqdm
import nibabel as nib


warnings.filterwarnings("ignore")



class H5PY_Supervised_SlicesData(Dataset): 
    def __init__(self, dataset_name, root=None, validation=False, test=False,seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        # Supervised branch; Total vol: 80
        if dataset_name == 'IXI':
            train_data, test_data  = IXI_T1_motion.get_ixi_t1_motion_h5py()
            self.getslicefromh5py = IXI_T1_motion.getslicefromh5py

        if validation:
            test_data = test_data[0:1]
            
        if root == None:
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data

        self.examples = []
        print('Loading dataset :', root)
        random.seed(seed)

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['still']

            # start_slice = np.random.randint(140, 180)
            # end_slice = start_slice + 10

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            # self.examples += [(fname, shotname, slice) for slice in range(start_slice, end_slice)] 
            self.examples += [(fname, shotname, slice) for slice in range(1, num_slices-1)] 

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        if test:
            self.transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            ) 
            self.target_transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            )           
        else:
            self.transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            ) 
            self.target_transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(15),
                    transforms.RandomHorizontalFlip(p=0.5)
                    
                ]
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):

        fname, shotname, slice = self.examples[i]
        still, with_motion = self.getslicefromh5py(fname, slice)

        still_down, with_motion_down = self.getslicefromh5py(fname, slice-1)
        still_up, with_motion_up = self.getslicefromh5py(fname, slice+1)

        seed = np.random.randint(2147483647) 
        still = self._process_img(still, seed)
        with_motion = self._process_img(with_motion, seed)

        with_motion_down = self._process_img(with_motion_down, seed)
        with_motion_up = self._process_img(with_motion_up, seed)

        return still, with_motion, shotname, with_motion_down, with_motion_up
    
    def _process_img(self, img, seed):
        img = img.astype(np.float32)
        img = 1.0 * (img - np.min(img)) / (np.max(img) - np.min(img))
        random.seed(seed) 
        torch.manual_seed(seed) 
        img = self.transform(img)
        return img


class H5PY_Unsupervised_SlicesData(Dataset): 
    def __init__(self, dataset_name, root=None, validation=False, test=False,seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        # Supervised branch; Total vol: 80
        if dataset_name == 'OASI1_MRB':
            train_data, test_data  = OASI1_MRB.get_oasi1mrb_edge_h5py()
            self.getslicefromh5py_Unsup = OASI1_MRB.getslicefromh5py
        elif dataset_name == 'MR_ART':
            train_data, test_data  = MR_ART.get_mr_art_h5py()
            self.getslicefromh5py_Unsup = MR_ART.getslicefromh5py

        if validation:
            test_data = test_data[0:1]
            
        if root == None:
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data



        self.examples = []

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['standard']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples += [(fname, shotname, slice) for slice in range(1, num_slices-1)] 

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        if test:
            self.transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            ) 
            self.target_transform = transforms.Compose([
                    transforms.ToTensor()

                ]
            )           
        else:
            self.transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(5),
                    transforms.RandomHorizontalFlip(p=0.1)
                    
                ]
            ) 
            self.target_transform = transforms.Compose([
                    
                    transforms.ToTensor(),
                    transforms.RandomRotation(5),  # csmri 15
                    transforms.RandomHorizontalFlip(p=0.1) # csmri 0.5
                    
                ]
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, shotname, slice = self.examples[i]
        headmotion1, headmotion2, standard = self.getslicefromh5py_Unsup(fname, slice)

        headmotion1_down, headmotion2_down, standard_down = self.getslicefromh5py_Unsup(fname, slice-1)
        headmotion1_up, headmotion2_up, standard_up = self.getslicefromh5py_Unsup(fname, slice+1)

        # headmotion1 = headmotion1.astype(np.float32)
        seed = np.random.randint(2147483647)
        headmotion2 = self._process_img(headmotion2, seed)
        standard = self._process_img(standard, seed)

        headmotion2_down = self._process_img(headmotion2_down, seed)
        headmotion2_up = self._process_img(headmotion2_up, seed)

        return standard, headmotion2, shotname, headmotion2_down, headmotion2_up

    def _process_img(self, img, seed):
        img = img.astype(np.float32)
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.resize(img, [240, 240])
        img = 1.0 * (img - np.min(img)) / (np.max(img) - np.min(img))
        random.seed(seed) 
        torch.manual_seed(seed) 
        img = self.transform(img)
        return img
    
if __name__ == '__main__':
    import time
    dataset_name_lists = ['IXI', 'MR_ART', 'OASI1_MRB']
    batch = 1
    # dataset = MiccaiSliceData() # 
    # dataset = MemoryMiccaiSliceData() # 很卡。
    dataset_sup = H5PY_Supervised_SlicesData(dataset_name='IXI')
    dataloader_sup = DataLoader(dataset_sup, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    dataset_unsup = H5PY_Unsupervised_SlicesData(dataset_name='MR_ART')
    dataloader_unsup = DataLoader(dataset_unsup, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    step = 0
    plotter = VisdomLinePlotter(env_name='main')

    loop = tqdm(enumerate(zip(dataloader_sup, dataloader_unsup)), total=len(dataloader_sup))

    for index, x in loop:
    # for x in dataloader:
        step += 1

        still, with_motion, shotname, with_motion_down, with_motion_up = x[0]
        standard, headmotion2, shotname, headmotion2_down, headmotion2_up = x[1]

        plotter.image(shotname[0][0:5]+'img1_still', step, still)
        plotter.image(shotname[0][0:5]+'img1_with_motion', step, with_motion)
        plotter.image(shotname[0][0:5]+'img2_headmotion2', step, headmotion2)
        plotter.image(shotname[0][0:5]+'img2_standard', step, standard)

        plotter.image(shotname[0][0:5]+'with_motion_down', step, with_motion_down)
        plotter.image(shotname[0][0:5]+'with_motion_up', step, with_motion_up)
        plotter.image(shotname[0][0:5]+'headmotion2_down', step, headmotion2_down)
        plotter.image(shotname[0][0:5]+'headmotion2_up', step, headmotion2_up)

        time.sleep(0.1)