import os
import time
import sys
import warnings
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rotate as torch_rotate
import yaml

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim
from kornia.filters import sobel

# model setting
sys.path.append('.') # 
from data.dataset import get_h5py_supervised_dataset, get_h5py_unsupervised_dataset
from model_backbones.recon_net import Unet, Hand_Tailed_Mask_Layer
# from model_backbones.drt import DeepRecursiveTransformer
# from model_backbones.restormer import Restormer

# from model_backbones.mist_plus_plus import MST_Plus_Plus  # nice transformer, but slow
from model_backbones.nafnet import NAFNet  # nice net, but overfitting
# from model_backbones.stripformer import Stripformer # nice transformer. but slow.
# from model_backbones.lama import FFCResNetGenerator


# from model_backbones.ffl_loss import FocalFrequencyLoss
from layers.vgg_loss import VGGPerceptualLoss
# from solver.loss_function import LocalStructuralLoss
from utils.visdom_visualizer import VisdomLinePlotter
from utils.train_utils import return_data_ncl_imgsize
from utils.train_metrics import Metrics, TVLoss
warnings.filterwarnings("ignore")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True



class ModelSet():
    def __init__(self, args, test=False):
        '''

        '''

        file_name = os.path.basename(__file__)
        module = file_name

        # self.nclasses, self.inputs_size = return_data_ncl_imgsize(args.dataset_name)
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()

        self.ckpt = './model_zoo/exp1/{}_{}_{}_{}.pth'.format(args.supervised_dataset_name, args.unsupervised_dataset_name, module, str(args.headmotion))
        args.save_path = self.ckpt
        # self.model = Unet().to(device, dtype=torch.float)
        self.sample_net = Hand_Tailed_Mask_Layer(0.33, 'cartesian', (240, 240), resize_size=[240,240]).to(device, dtype=torch.float)
        # self.model = Unet().to(device, dtype=torch.float)

        img_channel = 1
        width = 32
        # enc_blks = [2, 2, 4, 8]
        # middle_blk_num = 12
        # dec_blks = [2, 2, 2, 2]
        enc_blks = [1, 1, 1, 28]
        middle_blk_num = 1
        dec_blks = [1, 1, 1, 1]
        self.model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                        enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

        # self.model = Stripformer()
        # config_path = "./config/ffc_resnet075.yaml"
        # f = open(config_path, "r")
        # y = yaml.safe_load(f)



        # self.model = FFCResNetGenerator(1, 1, init_conv_kwargs=y["init_conv_kwargs"], downsample_conv_kwargs=y["downsample_conv_kwargs"], 
        #                         resnet_conv_kwargs=y["resnet_conv_kwargs"],
        #                         #  spatial_transform_layers=None, spatial_transform_kwargs=y.spatial_transform_kwargs,
        #                         #  add_out_act=True, max_features=1024, out_ffc=False, out_ffc_kwargs={}
        #                         )


        self.lr = args.lr  # batch up 2 and lr up 2: batch:lr = 4:0.001; one GPU equals 12 batch. /// 100epoch ~= 1hour.  
        self.num_epochs = args.epochs  # ! for MRB

        # print('MTL weight_recon={}, weight_seg={}, weight_local={}'.format(self.weight_recon, self.weight_seg, self.weight_local))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print('different lr0 = {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))

        patience = 100
        self.batch = args.batch_size
        self.save_epoch = 10  # 保存最好的模型
        self.milestones = [100, 120, 140, 160, 180, 200]
        # self.milestones = [200, 400]
        

        if test:
            self.model.load_state_dict(
                torch.load(args.save_path))
            print("Finished load model parameters! in {}".format(args.save_path))

        self.dataset_sup, self.val_loader_sup = get_h5py_supervised_dataset(args.supervised_dataset_name)
        self.dataset_unsup, self.val_loader_unsup = get_h5py_unsupervised_dataset(args.unsupervised_dataset_name)
        # lr decay: https://zhuanlan.zhihu.com/p/93624972 pytorch必须掌握的的4种学习率衰减策略
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        else:
            print( 'Using '+ str(device))
        self.model.to(device, dtype=torch.float)
        

        scheduler_trick = 'MultiStepLR'
        if scheduler_trick == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                        factor=0.1,
                                                        patience=patience, # epoch 放在外面
                                                        verbose=True,
                                                        threshold=0.0001,
                                                        min_lr=1e-8,
                                                        cooldown=4)
        elif scheduler_trick == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
        elif scheduler_trick == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                milestones=self.milestones, gamma=0.1) 
                       
        self.scheduler_trick = scheduler_trick
        print('scheduler_trick={}'.format(self.scheduler_trick))

        print('batch={}, lr={}, dataset_name={}_{}, load_model={}, save_per_epoch={}'\
            .format(self.batch, self.lr, args.supervised_dataset_name, args.unsupervised_dataset_name, args.load_model, 'best-model'))
        if test:
            print('It just a test' + '*'*20)

    def recon_criterion(self, out_recon, full_img):
        rec_loss = self.criterion(out_recon, full_img)
        return rec_loss

    def sobel_loss(self, recon, full_img):
        loss = nn.SmoothL1Loss()(sobel(recon), sobel(full_img))
        return loss

class Trainer:
    def __init__(self, dataset_sup: DataLoader, dataset_unsup: DataLoader, modelset: ModelSet):
        # Deblur GAN https://github.com/VITA-Group/DeblurGANv2/blob/master/train.py
        self.dataset_sup = dataset_sup
        self.dataset_unsup = dataset_unsup
        self.val_loader_sup = modelset.val_loader_sup
        self.val_loader_unsup = modelset.val_loader_unsup 
        self.modelset = modelset
        
        file_name = os.path.basename(__file__)
        self.plotter = VisdomLinePlotter(env_name=file_name)
        self.metrics = Metrics()

        self.num_epochs = modelset.num_epochs
        self.optimizer = modelset.optimizer
        self.scheduler = modelset.scheduler
        self.sample_net = modelset.sample_net
        self.model = modelset.model
        self.criterion = modelset.criterion
        self.sobel_loss = modelset.sobel_loss
        self.tv_loss = TVLoss()
        # self.ffloss = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0)
        self.vgg_loss = VGGPerceptualLoss().cuda()
        
    def train(self):
        # visdom
        dataloader_sup = self.dataset_sup
        dataloader_unsup = self.dataset_unsup

        save_path = args.save_path

        if os.path.exists(
                save_path) and args.load_model:
            self.model.load_state_dict(
                torch.load(save_path))
            print("Finished load model parameters!")
            print(save_path)
        epoch = 0
        for epoch in range(1, self.num_epochs+1):
            print("acc_rate: {}, dataset: ({}, {}), backbone: {}".format(args.headmotion, args.supervised_dataset_name, args.unsupervised_dataset_name, 'fix-me'))
            # print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 40)

            self._run_epoch_csmri(epoch, dataloader_sup)
            # self._run_epoch_sup_motion(epoch, dataloader_sup)
            # self._run_epoch_unsup(epoch, dataloader_unsup)

            # self._run_epoch_unsup_ei(epoch, dataloader_sup)

            self.scheduler.step()  # for step
            if epoch > 5:
                torch.save(self.model.state_dict(), save_path)
            if epoch%5 != 0:
                continue

            # self._validate(epoch, self.val_loader_sup)
            self._validate(epoch, self.val_loader_unsup)

    def _run_epoch_csmri(self, epoch, dataloader):
        step = 0
        epoch_loss = 0
        self.metrics.ssims, self.metrics.psnrs = [], []
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr'] # ! parameter group 0, group 1, ...

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for index, (still, with_motion, _shotname) in loop:
            step += 1
            
            # x_train = with_motion.to(device, dtype=torch.float)
            x_train = still.to(device, dtype=torch.float)
            y_train = still.to(device, dtype=torch.float)

            out = self.sample_net(x_train)
            x_train = out['complex_abs']

            x_train_rec = self.model(x_train)

            loss = self.criterion(x_train_rec, y_train)  + self.sobel_loss(x_train_rec, y_train) \
                + 1e-5*self.vgg_loss(x_train_rec, y_train, feature_layers=[0, 1, 2, 3], style_layers=[])

            # loss = self.criterion(x_train_rec, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            img_x = y_train.detach().cpu().numpy()[0, 0, ...]
            img_out = x_train_rec.detach().cpu().numpy()[0, 0, ...]

            ssim = compare_ssim(img_x, img_out, data_range=1)
            psnr = compare_psnr(img_x, img_out, data_range=1)
            self.metrics.get_metrics(psnr, ssim)

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=epoch_loss/step, psnr=self.metrics.mean_psnr, ssim=self.metrics.mean_ssim, lr=learning_rate)
            # loop.set_postfix(loss=epoch_loss/step, psnr=metrics.mean_psnr, ssim=metrics.mean_ssim, lr=learning_rate)

            if step == 1:

                self.plotter.image('still_sup', epoch, y_train)
                self.plotter.image('with_motion', epoch, x_train)
                self.plotter.image('rec_sup', epoch, x_train_rec)

                self.plotter.plot_epoch('loss', 'train', 'sup_Loss', epoch, epoch_loss/step)
                self.plotter.plot_epoch('PSNR', 'train', 'sup_PSNR', epoch, self.metrics.mean_psnr)
                self.plotter.plot_epoch('SSIM', 'train', 'sup_SSIM', epoch, self.metrics.mean_ssim)
        loop.close()

    def _run_epoch_unsup(self, epoch, dataloader):
        step = 0
        epoch_loss = 0
        self.metrics.ssims, self.metrics.psnrs = [], []
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr'] # ! parameter group 0, group 1, ...

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for index, (still, with_motion, _shotname) in loop:
            step += 1
            
            x_train = with_motion.to(device, dtype=torch.float)
            y_train = still.to(device, dtype=torch.float)

            x_train_rec = self.model(x_train)

            # loss = 1e-4 * (self.tv_loss(x_train_rec)  + self.ffloss(x_train_rec))
            loss = 1e-6 * self.tv_loss(x_train_rec)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            img_x = y_train.detach().cpu().numpy()[0, 0, ...]
            img_out = x_train_rec.detach().cpu().numpy()[0, 0, ...]

            ssim = compare_ssim(img_x, img_out, data_range=1)
            psnr = compare_psnr(img_x, img_out, data_range=1)
            self.metrics.get_metrics(psnr, ssim)

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=epoch_loss/step, psnr=self.metrics.mean_psnr, ssim=self.metrics.mean_ssim, lr=learning_rate)
            # loop.set_postfix(loss=epoch_loss/step, psnr=metrics.mean_psnr, ssim=metrics.mean_ssim, lr=learning_rate)

            if step == 1:

                self.plotter.image('still_unsup', epoch, y_train)
                self.plotter.image('with_motion_un', epoch, x_train)
                self.plotter.image('rec_unsup', epoch, x_train_rec)

                self.plotter.plot_epoch('loss', 'train', 'unsup_Loss', epoch, epoch_loss/step)
                self.plotter.plot_epoch('PSNR', 'train', 'unsup_PSNR', epoch, self.metrics.mean_psnr)
                self.plotter.plot_epoch('SSIM', 'train', 'unsup_SSIM', epoch, self.metrics.mean_ssim)
        loop.close()

    def _run_epoch_unsup_ei(self, epoch, dataloader):
        step = 0
        epoch_loss = 0
        self.metrics.ssims, self.metrics.psnrs = [], []
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr'] # ! parameter group 0, group 1, ...

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for index, (still, with_motion, _shotname) in loop:
            step += 1
            
            x_train = with_motion.to(device, dtype=torch.float)
            y_train = still.to(device, dtype=torch.float)

            out = self.sample_net(y_train)
            x_train = out['complex_abs']
            x_train_rec = self.model(x_train)
            loss1 = self.criterion(x_train_rec, y_train) +  self.sobel_loss(x_train_rec, y_train) \
                      +  1e-3*self.vgg_loss(x_train_rec, y_train, feature_layers=[0, 1, 2, 3], style_layers=[])# 1

            rotate_ytrain = torch_rotate(y_train, 90)
            out = self.sample_net(rotate_ytrain)

            x_train = out['complex_abs']
            x_train_rec = self.model(x_train)

            # loss = 1e-5 * self.tv_loss(x_train_rec)
            loss2 =self.criterion(x_train_rec, rotate_ytrain) +  self.sobel_loss(x_train_rec, rotate_ytrain) \
                    + 1e-3*self.vgg_loss(x_train_rec, rotate_ytrain, feature_layers=[0, 1, 2, 3], style_layers=[])

            loss = 1e-1 * (loss1 + loss2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            img_x = y_train.detach().cpu().numpy()[0, 0, ...]
            img_out = x_train_rec.detach().cpu().numpy()[0, 0, ...]

            ssim = compare_ssim(img_x, img_out, data_range=1)
            psnr = compare_psnr(img_x, img_out, data_range=1)
            self.metrics.get_metrics(psnr, ssim)

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=epoch_loss/step, psnr=self.metrics.mean_psnr, ssim=self.metrics.mean_ssim, lr=learning_rate)
            # loop.set_postfix(loss=epoch_loss/step, psnr=metrics.mean_psnr, ssim=metrics.mean_ssim, lr=learning_rate)

            if step == 1:

                self.plotter.image('still_ei', epoch, y_train)
                self.plotter.image('with_motion_ei', epoch, x_train)
                self.plotter.image('rec_ei', epoch, x_train_rec)

                self.plotter.plot_epoch('loss', 'train', 'unsup_Loss', epoch, epoch_loss/step)
                self.plotter.plot_epoch('PSNR', 'train', 'unsup_PSNR', epoch, self.metrics.mean_psnr)
                self.plotter.plot_epoch('SSIM', 'train', 'unsup_SSIM', epoch, self.metrics.mean_ssim)
        loop.close()
    def _run_epoch_sup_motion(self, epoch, dataloader):
        step = 0
        epoch_loss = 0
        self.metrics.ssims, self.metrics.psnrs = [], []
        learning_rate = self.optimizer.state_dict()['param_groups'][0]['lr'] # ! parameter group 0, group 1, ...

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for index, (still, with_motion, _shotname) in loop:
            step += 1
            
            # x_train = with_motion.to(device, dtype=torch.float)
            with_motion = with_motion.to(device, dtype=torch.float)
            y_train = still.to(device, dtype=torch.float)

            # out = self.sample_net(x_train)
            x_train = with_motion

            x_train_rec = self.model(x_train)

            loss = self.criterion(x_train_rec, y_train)  + self.sobel_loss(x_train_rec, y_train) \
                + 1e-5*self.vgg_loss(x_train_rec, y_train, feature_layers=[0, 1, 2, 3], style_layers=[])

            # loss = self.criterion(x_train_rec, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            img_x = y_train.detach().cpu().numpy()[0, 0, ...]
            img_out = x_train_rec.detach().cpu().numpy()[0, 0, ...]

            ssim = compare_ssim(img_x, img_out, data_range=1)
            psnr = compare_psnr(img_x, img_out, data_range=1)
            self.metrics.get_metrics(psnr, ssim)

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{self.num_epochs}]')
            loop.set_postfix(loss=epoch_loss/step, psnr=self.metrics.mean_psnr, ssim=self.metrics.mean_ssim, lr=learning_rate)
            # loop.set_postfix(loss=epoch_loss/step, psnr=metrics.mean_psnr, ssim=metrics.mean_ssim, lr=learning_rate)

            if step == 1:

                self.plotter.image('still_sup', epoch, y_train)
                self.plotter.image('with_motion', epoch, x_train)
                self.plotter.image('rec_sup', epoch, x_train_rec)

                self.plotter.plot_epoch('loss', 'train', 'sup_Loss', epoch, epoch_loss/step)
                self.plotter.plot_epoch('PSNR', 'train', 'sup_PSNR', epoch, self.metrics.mean_psnr)
                self.plotter.plot_epoch('SSIM', 'train', 'sup_SSIM', epoch, self.metrics.mean_ssim)
        loop.close()
    def _validate(self, epoch, val_loader):
        self.model.eval()
        with torch.no_grad():
            loss_val = 0
            ssim_val = 0
            psnr_val = 0
            test_num = 0
            val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
            for index_val, (still_test, withmotion_test, _shotname) in val_loop: #  增加GPU 利用率
                x_test = withmotion_test.to(device, dtype=torch.float)
                y_test = still_test.to(device, dtype=torch.float)

                x_test_rec = self.model(x_test)
                loss_val = self.criterion(x_test_rec, y_test)
                for i in range(0, y_test.shape[0]): # val dataloader batch
                    test_num += 1
                    img_x_test = y_test.detach().cpu().numpy()[i, 0, ...]
                    img_out_test = x_test_rec.detach().cpu().numpy()[i, 0, ...]
                    # max_value = max(np.max(img_x_test), np.max(img_out_test))
                    ssim_val += compare_ssim(img_x_test, img_out_test, data_range=1)
                    psnr_val += compare_psnr(img_x_test, img_out_test, data_range=1)
                    if test_num == 30:
                        self.plotter.image('val_recon', epoch, x_test_rec.detach().cpu())
                val_loop.set_description(f'Epoch [{epoch}/{self.num_epochs}]')
                val_loop.set_postfix(loss=loss_val.item()/y_test.shape[0], psnr=round(psnr_val/test_num, 2), ssim=round(100*ssim_val/test_num, 2))
            # print("Val:loss_val={}, psnr={}, ssim={}".format(loss_val.item(), round(psnr_val/test_num, 2), round(100*ssim_val/test_num, 2)))
            self.plotter.image('still_val', epoch, y_test)
            self.plotter.image('with_motion_val', epoch, x_test)
            self.plotter.image('rec_val', epoch, x_test_rec)
            
            self.plotter.plot_epoch('loss', 'val', 'Loss', epoch, loss_val.item()/y_test.shape[0])
            self.plotter.plot_epoch('PSNR', 'val', 'PSNR', epoch, round(psnr_val/test_num, 2))
            self.plotter.plot_epoch('SSIM', 'val', 'SSIM', epoch, round(100*ssim_val/test_num, 2))
        self.model.train()
        val_loop.close()

def train():
    # batch size and lr is default.
    modelset = ModelSet(args, test=args.test)

    dataloader_sup = DataLoader(modelset.dataset_sup, 
                            batch_size=modelset.batch, 
                            shuffle=True,
                            num_workers=4)  # 增加GPU利用率稳定性。
    dataloader_unsup = DataLoader(modelset.dataset_unsup, 
                            batch_size=modelset.batch, 
                            shuffle=True,
                            num_workers=4)  # 增加GPU利用率稳定性。
    
    trainer = Trainer(dataloader_sup, dataloader_unsup, modelset)
    trainer.train()
    # train_model()

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3, help="The learning rate") # 5e-4, 5e-5, 5e-6
    parser.add_argument(
        "--batch_size", type=int, default=32, help="The batch size")
    parser.add_argument(
        "--epochs", type=int, default=400, help="The weighted parameter") #

    parser.add_argument(
        "--headmotion",
        type=str,
        default=None,
        help="Slight, Heavy")
    parser.add_argument(
        "--supervised_dataset_name",
        type=str,
        default=None,
        help="which dataset")
    parser.add_argument(
        "--unsupervised_dataset_name",
        type=str,
        default=None,
        help="which dataset")
    parser.add_argument(
        "--load_model",
        type=int,
        default=0,
        choices=[0,1],
        help="reload last parameter?")

    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        choices=[True, False])
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="Axial, Coronal, Sagittal")
    
    args = parser.parse_args()
    train()