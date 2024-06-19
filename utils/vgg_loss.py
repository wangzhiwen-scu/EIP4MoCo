import torch
import numpy as np
from torch import Variable


def get_loss(self, fakeIm, realIm):
    # 生成图像
    f_fake = torch.from_numpy(np.zeros((8,3,256,256)))    # 构建3通道数组并转换为Tensor
    f_fake[:,0,:,:] = np.squeeze(fakeIm.data)             # 赋值
    f_fake[:,1,:,:] = np.squeeze(fakeIm.data)
    f_fake[:,2,:,:] = np.squeeze(fakeIm.data)
    f_fake = Variable(f_fake.cuda()).float()              # 对赋值后的Tensor首先转换为cuda，再转换为Variable，最后转换为Float类型
    f_fake = self.contentFunc.forward(f_fake)             # 输入到trained network中

    # 目标图像
    f_real = torch.from_numpy(np.zeros((8,3,256,256)))    # 同上
    # f_real.cuda()
    f_real[:,0,:,:] = np.squeeze(realIm.data)
    f_real[:,1,:,:] = np.squeeze(realIm.data)
    f_real[:,2,:,:] = np.squeeze(realIm.data)
    f_real = Variable(f_real.cuda()).float()
    f_real = self.contentFunc.forward(f_real)

    f_real_no_grad = f_real.detach()
    loss = self.criterion(f_fake, f_real_no_grad)        # 计算感知损失
    return loss