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
from data.bezier_curve import nonlinear_transformation

warnings.filterwarnings("ignore")


class H5PYMixedSliceData(Dataset): 
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
        # elif dataset_name == 'IXI': 
        train_data_sup, test_data_sup  = IXI_T1_motion.get_ixi_t1_motion_h5py()
        self.getslicefromh5py_Sup = IXI_T1_motion.getslicefromh5py
        paired_files_sup = train_data_sup

        self.examples_Sup = []
        print('Loading dataset :', root)
        random.seed(seed)

        for fname in paired_files_sup:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['still']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples_Sup += [(fname, shotname, slice) for slice in range(num_slices)] 

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        
        # Unsupervised branch, Total vol: 80, same with Supervised branch.
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



        self.examples_Unsup = []

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['standard']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples_Unsup += [(fname, shotname, slice) for slice in range(num_slices)] 

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
        return len(self.examples_Sup)

    def __getitem__(self, i):

        fname, shotname, slice = self.examples_Sup[i]
        still, with_motion = self.getslicefromh5py_Sup(fname, slice)

        fname, shotname, slice = self.examples_Unsup[i]
        headmotion1, headmotion2, standard = self.getslicefromh5py_Unsup(fname, slice)

        still = still.astype(np.float32)
        with_motion = with_motion.astype(np.float32)

        # headmotion1 = headmotion1.astype(np.float32)
        headmotion2 = headmotion2.astype(np.float32)
        standard = standard.astype(np.float32)

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 

        still = self.transform(still)
        with_motion = self.transform(with_motion)

        headmotion2 = self.transform(headmotion2)
        # still = self.transform(still)
        standard = self.transform(standard)

        return still, with_motion, headmotion2, standard, shotname


class H5PY_Supervised_Data(Dataset): 
    def __init__(self, dataset_name, change_isotopic=False, root=None, validation=False, test=False, tta_testing=False, bezier=False, seed=42):
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
        self.tta_dataset_name = dataset_name
        self.bezier = bezier
        if dataset_name == 'IXI':
            train_data, test_data_org  = IXI_T1_motion.get_ixi_t1_motion_h5py()
            self.getslicefromh5py = IXI_T1_motion.getslicefromh5py

        elif dataset_name[:3] == 'ixi' or dataset_name[:3] == 'sta' or dataset_name[:3] == 'fas' or dataset_name[:3] == 'mrb' or \
            dataset_name[:4] == 'andi':
            train_data, test_data_org = IXI_T1_motion.get_ixi_t1_motion_h5py(subdir=dataset_name)
            if dataset_name == 'ixi_t1_periodic_slight_coronal_xxx': # out-of-fasion
                self.getslicefromh5py = IXI_T1_motion.getslicefromh5py_coronal # [:,slice,:]
            elif dataset_name == 'ixi_t1_periodic_slight_sagittal_xxx':  # out-of-fasion
                self.getslicefromh5py = IXI_T1_motion.getslicefromh5py_sagittal # [:,:,slice]
            else:
                self.getslicefromh5py = IXI_T1_motion.getslicefromh5py # [slice,:,:]
        elif dataset_name[:5] == 'liver':
            train_data, test_data_org = IXI_T1_motion.get_respiratory_motion_h5py(subdir=dataset_name)
            self.getslicefromh5py = IXI_T1_motion.getslicefromh5py # [slice,:,:]
        elif dataset_name[:4] == 'acdc':
            train_data, test_data_org = IXI_T1_motion.get_cardiac_motion_h5py(subdir=dataset_name)
            self.getslicefromh5py = IXI_T1_motion.getsliceandframefromh5py # [slice,:,:]
            # if dataset_name != 'ixi_t1_periodic_slight':
            #     train_data = test_data_org
            # elif dataset_name == 'ixi_t1_periodic_slight' and tta_testing:
            #     train_data = test_data_org

            # if dataset_name != 'ixi_t1_periodic_slight':
                # train_data = test_data_org
                
        if tta_testing:
            train_data = test_data_org[5:10]

        if validation:

            test_data = test_data_org[5:10]
            # test_data = test_data_org
            
        if root == None:
            # paired_files = train_data[0:24]
            paired_files = train_data
        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data_org
            # paired_files = test_data
            # if dataset_name[:4] == 'acdc':
                # paired_files = paired_files[0:1]


        self.examples = []
        print('Loading dataset :', root)
        random.seed(seed)

        if paired_files: # for paired_files = []
            for fname in paired_files:
                h5f = h5py.File(fname, 'r')
                dataimg = h5f['still']

                # start_slice = np.random.randint(140, 180)
                # end_slice = start_slice + 10

                shotname = get_h5py_shotname(fname)
                if dataset_name == 'ixi_t1_periodic_slight_coronal_xxx':
                    num_slices = dataimg.shape[1]
                elif dataset_name == 'ixi_t1_periodic_slight_sagittal_xxx':
                    num_slices = dataimg.shape[2]
                else:
                    num_slices = dataimg.shape[0] 
                # self.examples += [(fname, shotname, slice) for slice in range(start_slice, end_slice)] 
                if dataset_name[:4] == 'acdc':
                    self.examples += [(fname, shotname, (frame, slice)) for frame in range(dataimg.shape[0]) for slice in range(dataimg.shape[1])]
                else:
                    self.examples += [(fname, shotname, slice) for slice in range(num_slices)] 

        # if test and dataset_name[:4] == 'acdc':
        #     self.examples = random.sample(self.examples, 300)

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        if test or not change_isotopic:
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

        still = still.astype(np.float32)
        # still = cv2.resize(still, [240, 240])
        # still = cv2.rotate(still, cv2.ROTATE_90_CLOCKWISE)
        still = 1.0 * (still - np.min(still)) / (np.max(still) - np.min(still))
        # still = still/(np.max(still) - np.min(still))

        # noise = np.random.normal(0, 0.02, still.shape)
        # with_noise = still + noise

        # with_noise = 1.0 * (with_noise - np.min(with_noise)) / (np.max(with_noise) - np.min(with_noise))
        # with_motion = with_noise

        with_motion = with_motion.astype(np.float32)
        # with_motion = cv2.rotate(with_motion, cv2.ROTATE_90_CLOCKWISE)
        # with_motion = cv2.resize(with_motion, [240, 240])
        with_motion = 1.0 * (with_motion - np.min(with_motion)) / (np.max(with_motion) - np.min(with_motion))
        # with_motion = with_motion/(np.max(with_motion) - np.min(with_motion))

        if self.tta_dataset_name[:6] == 'ixi_t2' or self.tta_dataset_name[:6] == "ixi_pd" \
            or self.tta_dataset_name == 'stanford_knee_axial_pd_periodic_slight' or self.tta_dataset_name[:15] == 'fastmribrain_t1' \
            or self.tta_dataset_name == 'mrb13_t1_sudden_sslight':

            still = cv2.resize(still, [240, 240])
            with_motion = cv2.resize(with_motion, [240, 240])
        if self.tta_dataset_name[:6] == 'ixi_t2' or self.tta_dataset_name[:6] == "ixi_pd":
            still = cv2.rotate(still, cv2.ROTATE_180)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)
        
        # if self.tta_dataset_name[:6] == 'ixi_t2' and self.bezier:
        if self.bezier:
            still = random.choice(nonlinear_transformation(still))
            with_motion = random.choice(nonlinear_transformation(with_motion))

        if self.tta_dataset_name == 'stanford_knee_axial_pd_periodic_moderate' or  \
             self.tta_dataset_name == 'ixi_t1_periodic_slight_rl':
            still = cv2.rotate(still, cv2.ROTATE_90_COUNTERCLOCKWISE)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.tta_dataset_name[:15] == 'fastmribrain_t1':
            still = cv2.rotate(still, cv2.ROTATE_180)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)

        # if self.tta_dataset_name[:4] == 'andi':
        #     still = cv2.rotate(still, cv2.ROTATE_180)
        #     with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)

            # still = cv2.resize(still, [240, 240])
            # with_motion = cv2.resize(with_motion, [240, 240])

        if self.tta_dataset_name == 'ixi_t1_periodic_slight_coronal' or self.tta_dataset_name == 'ixi_t1_periodic_slight_sagittal' or \
            self.tta_dataset_name == 'ixi_t1_periodic_moderate_coronal' or self.tta_dataset_name == 'ixi_t1_periodic_moderate_sagittal':
            still = cv2.rotate(still, cv2.ROTATE_180)
            with_motion = cv2.rotate(with_motion, cv2.ROTATE_180)
            
            still = cv2.flip(still, 1) # 1 is vertically flip
            with_motion = cv2.flip(with_motion, 1)

            # still = cv2.rotate(still, cv2.ROTATE_90_CLOCKWISE)
            # with_motion = cv2.rotate(with_motion, cv2.ROTATE_90_CLOCKWISE)

        if self.tta_dataset_name[:5] == 'liver':
            if still.shape == (290, 320):
                still = cv2.resize(still, [320, 288])
                with_motion = cv2.resize(with_motion, [320, 288])
            elif still.shape == (260, 320):
                still = cv2.resize(still, [320, 256])
                with_motion = cv2.resize(with_motion, [320, 256])
            elif still.shape == (210, 320):
                still = cv2.resize(still, [320, 240])
                with_motion = cv2.resize(with_motion, [320, 240])

        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        still = self.transform(still)
        random.seed(seed) 
        torch.manual_seed(seed) 
        with_motion = self.transform(with_motion)

        return still, with_motion, shotname

class H5PY_Unsupervised_Data(Dataset): 
    def __init__(self, dataset_name, change_isotopic=True, root=None, validation=False, test=False, slight_motion=False, one_shot=False, seed=42):
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
        self.slight_motion = slight_motion
        # Supervised branch; Total vol: 80
        self.change_isotopic = change_isotopic
        if dataset_name == 'OASI1_MRB':
            train_data, test_data  = OASI1_MRB.get_oasi1mrb_edge_h5py()
            self.getslicefromh5py_Unsup = OASI1_MRB.getslicefromh5py
        elif dataset_name == 'MR_ART':
            train_data, test_data_orig  = MR_ART.get_mr_art_h5py()
            self.getslicefromh5py_Unsup = MR_ART.getslicefromh5py
        elif dataset_name[:3] == 'ixi':
            train_data, test_data_orig = IXI_T1_motion.get_ixi_t1_motion_h5py(subdir=dataset_name)
            self.getslicefromh5py_Unsup = IXI_T1_motion.getslicefromh5py

        if validation:
            test_data = test_data_orig[0:1]
            
        if root == None:
            paired_files = train_data
            if one_shot:
                paired_files = train_data[0:1]

        if validation:
            paired_files = test_data
        if test:
            paired_files = test_data_orig



        self.examples = []

        for fname in paired_files:
            h5f = h5py.File(fname, 'r')
            dataimg = h5f['standard']

            shotname = get_h5py_shotname(fname)
            num_slices = dataimg.shape[0] 
            self.examples += [(fname, shotname, slice) for slice in range(num_slices)] 

        #! writ it in one examples.... https://blog.csdn.net/wunianwn/article/details/126965641
        if test or not change_isotopic:
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

        if self.slight_motion:
            headmotion2 = headmotion1
        # headmotion1 = headmotion1.astype(np.float32)
        headmotion2 = headmotion2.astype(np.float32)
        headmotion2 = cv2.rotate(headmotion2, cv2.ROTATE_180)
        if self.change_isotopic:
            # headmotion2 = cv2.resize(headmotion2, [384, 240])
            headmotion2 = self._get_240img(headmotion2)
        headmotion2 = 1.0 * (headmotion2 - np.min(headmotion2)) / (np.max(headmotion2) - np.min(headmotion2))
        if not self.change_isotopic:
            headmotion2 = self._get_240img(headmotion2)

        standard = standard.astype(np.float32)
        standard = cv2.rotate(standard, cv2.ROTATE_180)
        if self.change_isotopic:
            # standard = cv2.resize(standard, [384, 240])
            standard = self._get_240img(standard)

        standard = 1.0 * (standard - np.min(standard)) / (np.max(standard) - np.min(standard))
        if not self.change_isotopic:
            standard = self._get_240img(standard)
        seed = np.random.randint(2147483647) 
        random.seed(seed) 
        torch.manual_seed(seed) 
        headmotion2 = self.transform(headmotion2)
        # still = self.transform(still)
        random.seed(seed) 
        torch.manual_seed(seed) 
        standard = self.transform(standard)

        return standard, headmotion2, shotname

    def _get_240img(self, input_img):
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
        return output

if __name__ == '__main__':
    import time
    dataset_name_lists = ['IXI', 'MR_ART', 'OASI1_MRB']
    batch = 1
    # dataset = MiccaiSliceData() # 
    # dataset = MemoryMiccaiSliceData() # 很卡。
    dataset_sup = H5PY_Supervised_Data(dataset_name='IXI')
    dataloader_sup = DataLoader(dataset_sup, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    dataset_unsup = H5PY_Unsupervised_Data(dataset_name='MR_ART')
    dataloader_unsup = DataLoader(dataset_unsup, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)  # 增加GPU利用率稳定性。

    step = 0
    plotter = VisdomLinePlotter(env_name='main')

    # loop = tqdm(enumerate(zip(dataloader_sup, dataloader_unsup)), total=len(dataloader_sup))

    # for index, x in loop:
    for x in dataloader_sup:
        step += 1

        still, with_motion, shotname = x
        # standard, headmotion2, shotname = x[1]

        if [still.shape[2], still.shape[3]] != [240,240]:
            print(still.shape)
        if [with_motion.shape[2], with_motion.shape[3]] != [240,240]:
            print(with_motion.shape)

        # plotter.image(shotname[0][0:5]+'img1_still', step, still)
        # plotter.image(shotname[0][0:5]+'img1_with_motion', step, with_motion)
        # plotter.image(shotname[0][0:5]+'img2_headmotion2', step, headmotion2)
        # plotter.image(shotname[0][0:5]+'img2_standard', step, standard)

        # time.sleep(0.1)