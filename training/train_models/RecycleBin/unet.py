import os
import time
import sys
import warnings
import argparse
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

# model setting
sys.path.append('.') # 
from data.dataset import get_h5py_supervised_dataset, get_h5py_unsupervised_dataset
from model_backbones.recon_net import Unet
# from solver.loss_function import LocalStructuralLoss
from utils.visdom_visualizer import VisdomLinePlotter
from utils.train_utils import return_data_ncl_imgsize
from utils.train_metrics import Metrics
warnings.filterwarnings("ignore")

global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class ModelSet():
    def __init__(self, args, test=False):
        '''

        '''

        module = ''

        # self.nclasses, self.inputs_size = return_data_ncl_imgsize(args.dataset_name)
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))

        self.criterion = nn.SmoothL1Loss()
        # self.criterion = nn.MSELoss()

        self.ckpt = './model_zoo/exp1/unet_{}_{}_{}_{}.pth'.format(args.supervised_dataset_name, args.unsupervised_dataset_name, module, str(args.headmotion))
        args.save_path = self.ckpt
        self.model = Unet().to(device, dtype=torch.float)

        self.lr = args.lr  # batch up 2 and lr up 2: batch:lr = 4:0.001; one GPU equals 12 batch. /// 100epoch ~= 1hour.  
        self.num_epochs = args.epochs  # ! for MRB

        # print('MTL weight_recon={}, weight_seg={}, weight_local={}'.format(self.weight_recon, self.weight_seg, self.weight_local))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # print('different lr0 = {}'.format(self.optimizer.state_dict()['param_groups'][0]['lr']))

        patience = 100
        self.batch = args.batch_size
        self.save_epoch = 10  # 保存最好的模型
        # self.milestones = [100, 120, 140, 160, 180, 200]
        self.milestones = [200, 400]

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

        scheduler_trick = 'ReduceLROnPlateau'
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
    
#!######################################-seg-#########################################################

def train_model(dataloader_sup, dataloader_unsup, modelset):
    # visdom
    plotter = VisdomLinePlotter(env_name='unet')
    num_epochs = modelset.num_epochs
    optimizer = modelset.optimizer
    scheduler = modelset.scheduler
    model = modelset.model
    val_loader = modelset.val_loader_sup  #  为了

    criterion = modelset.criterion
    save_path = args.save_path

    if os.path.exists(
            save_path) and args.load_model:
        model.load_state_dict(
            torch.load(save_path))
        print("Finished load model parameters!")
        print(save_path)

    metrics = Metrics()
    min_loss = 10000
    ssim = 1
    epoch = 0

    for epoch in range(1, num_epochs+1):
        print("acc_rate: {}, dataset: ({}, {}), backbone: {}".format(args.headmotion, args.supervised_dataset_name, args.unsupervised_dataset_name, 'fix-me'))
        # print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 40)
        dt_size = len(dataloader_sup.dataset)
        epoch_loss = 0

        step = 0
        metrics.ssims, metrics.psnrs = [], []
        metrics.dices = []

        learning_rate = optimizer.state_dict()['param_groups'][0]['lr'] # ! parameter group 0, group 1, ...
        time_tik = time.time()

        # https://blog.csdn.net/wunianwn/article/details/126965641
        # for i, ((images, labels),rd) in enumerate(cycle(train_loader1,Rd_loader1)):

        loop = tqdm(enumerate(dataloader_sup), total=len(dataloader_sup))
        for index, (still, with_motion, _shotname) in loop:
            step += 1
            
            x_train = with_motion.to(device, dtype=torch.float)
            y_train = still.to(device, dtype=torch.float)

            x_train_rec = model(x_train)

            loss = criterion(x_train_rec, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            img_x = y_train.detach().cpu().numpy()[0, 0, ...]
            img_out = x_train_rec.detach().cpu().numpy()[0, 0, ...]
            img_org = x_train.detach().cpu().numpy()[0, 0, ...]

            max_value = max(np.max(img_x), np.max(img_out))
            max_value = np.max(img_x)
            ssim = compare_ssim(img_x, img_out, data_range=1)
            psnr = compare_psnr(img_x, img_out, data_range=1)

            # mssim = compare_ssim(img_x, img_org, data_range=1)
            # mpsnr = compare_psnr(img_x, img_org, data_range=1)
            # dice = dice_coef(img_seg, img_seglabel)
            metrics.get_metrics(psnr, ssim)

            # print('\r Epoch={}, Step={}/{}, psnr={}, ssim={}'.format(epoch, step, (dt_size - 1) // dataload.batch_size + 1, 
            # metrics.psnrs_mean_var, metrics.ssims_mean_var), end='', flush=True)

            # update tqdm
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=epoch_loss/step, psnr=metrics.mean_psnr, ssim=metrics.mean_ssim, lr=learning_rate)
            # loop.set_postfix(loss=epoch_loss/step, psnr=metrics.mean_psnr, ssim=metrics.mean_ssim, lr=learning_rate)

            if step == 1:

                plotter.image('still_sup', epoch, y_train)
                plotter.image('with_motion', epoch, x_train)
                plotter.image('rec_sup', epoch, x_train_rec)

                plotter.plot_epoch('loss', 'train', 'sup_Loss', epoch, epoch_loss/step)
                plotter.plot_epoch('PSNR', 'train', 'sup_PSNR', epoch, metrics.mean_psnr)
                plotter.plot_epoch('SSIM', 'train', 'sup_SSIM', epoch, metrics.mean_ssim)

        if modelset.scheduler_trick == 'ReduceLROnPlateau':
            scheduler.step(metrics=epoch_loss/step)  # for plateau
        else:
            scheduler.step()  # for step
        time_elapsed = time.time() - time_tik
        # print(" lr0={}, stage={}, time_elapsed={}".format(learning_rate, args.stage, time_elapsed), end='\n')
            
        # if epoch_loss/step < min_loss:
        #     min_loss = epoch_loss/step
        #     print('Current min_loss=:{}'.format(min_loss), end='\n')
                # evaluate model

        if epoch > 5:
            # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            torch.save(model.state_dict(), save_path)

        if epoch%5 != 0:
            continue

        model.eval()
        with torch.no_grad():
            loss_val = 0
            ssim_val = 0
            psnr_val = 0
            test_num = 0
            val_loop = tqdm(enumerate(val_loader), total=len(val_loader))
            for index_val, (still_test, withmotion_test, _shotname) in val_loop: #  增加GPU 利用率
                x_test = withmotion_test.to(device, dtype=torch.float)
                y_test = still_test.to(device, dtype=torch.float)

                x_test_rec = model(x_test)
                loss_val = criterion(x_test_rec, y_test)
                for i in range(0, y_test.shape[0]): # val dataloader batch
                    test_num += 1
                    img_x_test = y_test.detach().cpu().numpy()[i, 0, ...]
                    img_out_test = x_test_rec.detach().cpu().numpy()[i, 0, ...]
                    max_value = max(np.max(img_x_test), np.max(img_out_test))
                    ssim_val += compare_ssim(img_x_test, img_out_test, data_range=1)
                    psnr_val += compare_psnr(img_x_test, img_out_test, data_range=1)
                    if test_num == 30:
                        plotter.image('val_recon', epoch, x_test_rec.detach().cpu())
                val_loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
                val_loop.set_postfix(loss=loss_val.item()/y_test.shape[0], psnr=round(psnr_val/test_num, 2), ssim=round(100*ssim_val/test_num, 2))
            # print("Val:loss_val={}, psnr={}, ssim={}".format(loss_val.item(), round(psnr_val/test_num, 2), round(100*ssim_val/test_num, 2)))
            plotter.plot_epoch('loss', 'val', 'Loss', epoch, loss_val.item()/y_test.shape[0])
            plotter.plot_epoch('PSNR', 'val', 'PSNR', epoch, round(psnr_val/test_num, 2))
            plotter.plot_epoch('SSIM', 'val', 'SSIM', epoch, round(100*ssim_val/test_num, 2))

        model.train()


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

    train_model(dataloader_sup, dataloader_unsup, modelset)

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
        help="The type of mask")
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
    args = parser.parse_args()
    train()