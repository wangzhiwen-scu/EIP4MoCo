from torch import nn
import torch
from torch import distributed as dist
from collections import OrderedDict
import torch.nn.functional as F

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: n able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relatiumber of object categories, omitting the special no-object category
            matcher: moduleve classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        for k in self.losses.keys():
            # k, loss = loss_dict
            if k == 'loss':
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts#self._parse_losses(self.loss_dicts)

    def _parse_losses(self, losses):
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # top-1, top-5 both belong to accuracy
            elif isinstance(loss_value, dict):
                # log_vars[loss_name] = {}
                for name, value in loss_value.items():
                    log_vars[name] = value
                    # log_vars[loss_name].update({name: value.item()})
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        assert 'loss' not in log_vars.keys(), KeyError("key: 'loss' can't be set from cfg_file.")
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'top' not in _key)#if 'loss' in _key
        log_vars['loss'] = loss
        # output = log_vars.pop('acc') #get
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        # log_vars.update(acc=output)

        return loss, log_vars
    
class PatchMergeModule(nn.Module):

    def __init__(self, args=None, bs_axis_merge=False):
        super().__init__()
        if bs_axis_merge:
            self.split_func = torch.split
        else:
            self.split_func = lambda x, _, dim: [x]
        self.args = args

    def forward(self, x):

        return x, x

    def forward_chop(self, *x, shave=12, **kwargs):
        # 不存在输入张量不一样的情况, 如不一样请先处理成一样的再输入
        # 但输出存在维度不一样的情况, 因为网络有多层并且每层尺度不同, 进行分开处理: final_output, intermediate_output
        # TODO: bug: 1. 输出的张量尺寸不一样的时候 无法进行fold操作, 一是因为参数, 二是可能还原不回去
        #            2. unfold的在重叠切的时候是共享原始数据的，所以后者会覆盖前者

        x = torch.cat(x, dim=0)
        split_func = self.split_func
        args = self.args
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        x.cpu()
        batchsize = args.crop_batch_size
        patch_size = args.patch_size

        b, c, h, w = x.size()
        if isinstance(patch_size, (list, tuple)):
            (h_padsize, w_padsize) = patch_size
            hshave, wshave = h_padsize // 2, w_padsize // 2
        else:
            h_padsize = w_padsize = int(patch_size)
            hshave = wshave = patch_size // 2
        # print(self.scale, self.idx_scale)
        scale = args.scale[0]  # self.scale[self.idx_scale]

        h_cut = (h - h_padsize) % (int(hshave / 2))
        w_cut = (w - w_padsize) % (int(wshave / 2))

        x_unfold = F.unfold(x, (h_padsize, w_padsize), stride=(int(hshave / 2), wshave // 2)).permute(2, 0,
                                                                                                      1).contiguous()

        ################################################
        # 最后一块patch单独计算
        ################################################

        x_hw_cut = x[..., (h - h_padsize):, (w - w_padsize):]
        y_hw_cut = self.forward(*[s.cuda() for s in split_func(x_hw_cut, [1, 1], dim=0)], **kwargs)

        x_h_cut = x[..., (h - h_padsize):, :]
        x_w_cut = x[..., :, (w - w_padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
                             **kwargs)
        y_w_cut = self.cut_w(x_w_cut, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
                             **kwargs)
        # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 左上patch单独计算，不是平均而是覆盖
        ################################################

        x_h_top = x[..., :h_padsize, :]
        x_w_top = x[..., :, :w_padsize]
        y_h_top = self.cut_h(x_h_top, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
                             **kwargs)
        y_w_top = self.cut_w(x_w_top, h, w, c, h_cut, w_cut, (h_padsize, w_padsize), (hshave, wshave), scale, batchsize,
                             **kwargs)

        # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # img->patch，最大计算crop_s个patch，防止bs*p*p太大
        ################################################

        x_unfold = x_unfold.view(x_unfold.size(0), -1, c, h_padsize, w_padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            res = self.forward(
                *[s[:, 0, ...] for s in split_func(x_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)],
                **kwargs)
            y_unfold.append(res)
            # y_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...] for s in split_func(x_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)]
            #     , **kwargs)])
            # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())

        # for i, s in enumerate(zip(*y_unfold)):
        #     if i < len(y_unfold):
        #         y_unfold[i] = s
        #     else:
        #         y_unfold.append(s)
        if isinstance(y_unfold[0], tuple):
            y_unfold_out = [None] * len(y_unfold[0])
            for i, s in enumerate(zip(*y_unfold)):
                y_unfold_out[i] = s
            y_unfold = y_unfold_out
            y_hw_cut = [s.cpu() for s in y_hw_cut]
        else:
            y_unfold = [y_unfold]
            y_hw_cut = [y_hw_cut.cpu()]

        out = []
        for s_unfold, s_h_top, s_w_top, s_h_cut, s_w_cut, s_hw_cut in zip(y_unfold, y_h_top, y_w_top, y_h_cut, y_w_cut,
                                                                          y_hw_cut):
            s_unfold = torch.cat(s_unfold, dim=0).cpu()

            y = F.fold(s_unfold.view(s_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                       ((h - h_cut) * scale, (w - w_cut) * scale), (h_padsize * scale, w_padsize * scale),
                       stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)))
            # 312， 480
            # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            ################################################
            # 第一块patch->y
            ################################################
            y[..., :h_padsize * scale, :] = s_h_top
            y[..., :, :w_padsize * scale] = s_w_top
            # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            s_unfold = s_unfold[...,
                       int(hshave / 2 * scale):h_padsize * scale - int(hshave / 2 * scale),
                       int(wshave / 2 * scale):w_padsize * scale - int(wshave / 2 * scale)].contiguous()

            s_inter = F.fold(s_unfold.view(s_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                             ((h - h_cut - hshave) * scale, (w - w_cut - wshave) * scale),
                             (h_padsize * scale - hshave * scale, w_padsize * scale - wshave * scale),
                             stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)))
            # 1，3，750，540
            #
            s_ones = torch.ones(s_inter.shape, dtype=s_inter.dtype)
            divisor = F.fold(F.unfold(s_ones, (h_padsize * scale - hshave * scale, w_padsize * scale - wshave * scale),
                                      stride=(int(hshave / 2 * scale), int(wshave / 2 * scale))),
                             ((h - h_cut - hshave) * scale, (w - w_cut - wshave) * scale),
                             (h_padsize * scale - hshave * scale, w_padsize * scale - wshave * scale),
                             stride=(int(hshave / 2 * scale), int(wshave / 2 * scale)))

            s_inter = s_inter.cpu() / divisor
            # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            ################################################
            # 第一个半patch
            ################################################
            y[..., int(hshave / 2 * scale):(h - h_cut) * scale - int(hshave / 2 * scale),
            int(wshave / 2 * scale):(w - w_cut) * scale - int(wshave / 2 * scale)] = s_inter
            # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            y = torch.cat([y[..., :y.size(2) - int((h_padsize - h_cut) / 2 * scale), :],
                           s_h_cut[..., int((h_padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
            # 图分为前半和后半
            # x->y_w_cut
            # model->y_hw_cut TODO:check
            y_w_cat = torch.cat([s_w_cut[..., :s_w_cut.size(2) - int((h_padsize - h_cut) / 2 * scale), :],
                                 s_hw_cut[..., int((h_padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
            y = torch.cat([y[..., :, :y.size(3) - int((w_padsize - w_cut) / 2 * scale)],
                           y_w_cat[..., :, int((w_padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
            out.append(y.cuda())
            # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            # plt.show()

        return out  # y.cuda()

    def cut_h(self, x_h_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs):
        split_func = self.split_func
        # 1, 3, 30, 600 -> (30, 120), s=(7, 30)
        # 1,3*30*120, 17 -> 17, 1, 3, 30, 120
        # N = [(H - k_h + 2*pad_h) / s_h + 1] * [(W - k_w + 2*pad_w) / s_w + 1]
        #   = [1+(30 - 30) / 7] * [1+(600 - 120) / 30] = 1*17 = 17
        x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=(int(shave[0] / 2), int(shave[1] / 2))).permute(2, 0,
                                                                                                           1).contiguous()  # transpose(0, 2)
        # N, B, -1, c, ph, pw: 17, 1, 3, 30, 120
        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, c,
                                             *padsize)  # x_h_cut_unfold.size(0), -1, padsize, padsize
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()

        for i in range(x_range):
            res = self.forward(*[s[:, 0, ...] for s in
                                 split_func(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)],
                               **kwargs)
            y_h_cut_unfold.append(res)
            # y_h_cut_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...] for s in split_func(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize,
            #                                    ...], [1, 1], dim=1)], **kwargs)])  # P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())

        # [[a0, b0, c0], [a1, b1, c1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]

        if isinstance(y_h_cut_unfold[0], tuple):
            y_h_cut_unfold_out = [None] * len(y_h_cut_unfold[0])
            for i, s in enumerate(zip(*y_h_cut_unfold)):
                y_h_cut_unfold_out[i] = s
            y_h_cut_unfold = y_h_cut_unfold_out
        else:
            y_h_cut_unfold = [y_h_cut_unfold]
        y_h_cut = []

        for s_h_cut_unfold in y_h_cut_unfold:
            s_h_cut_unfold = torch.cat(s_h_cut_unfold, dim=0).cpu()
            # s_h_cut_unfold = s_h_cut_unfold.reshape(-1, c, *s_h_cut_unfold.size()[1:])
            # nH*nW, c, k, k: 3, 3, 100, 100 (17, 3, 30, 120)
            # out_size=(30, 600), k=(30, 120)
            s_h_cut = F.fold(
                s_h_cut_unfold.view(s_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                (padsize[0] * scale, (w - w_cut) * scale), (padsize[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
            s_h_cut_unfold = s_h_cut_unfold[..., :,
                             int(shave[1] / 2 * scale):padsize[1] * scale - int(shave[1] / 2 * scale)].contiguous()
            # 17, 3, 30, 60
            # out_size=(30, 540), k=(30, 90)
            s_h_cut_inter = F.fold(
                s_h_cut_unfold.view(s_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                (padsize[0] * scale, (w - w_cut - shave[1]) * scale),
                (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))

            s_ones = torch.ones(s_h_cut_inter.shape, dtype=s_h_cut_inter.dtype)
            divisor = F.fold(
                F.unfold(s_ones, (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
                         stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale))),
                (padsize[0] * scale, (w - w_cut - shave[1]) * scale),
                (padsize[0] * scale, padsize[1] * scale - shave[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
            s_h_cut_inter = s_h_cut_inter.cpu() / divisor

            s_h_cut[..., :, int(shave[1] / 2 * scale):(w - w_cut) * scale - int(shave[1] / 2 * scale)] = s_h_cut_inter
            y_h_cut.append(s_h_cut)

        return y_h_cut

    def cut_w(self, x_w_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs):

        split_func = self.split_func
        # 1, 3, 792, 120 -> (30, 120), s=(7, 30) -> 109, 1, 3, 30, 120
        x_w_cut_unfold = F.unfold(x_w_cut, padsize, stride=(int(shave[0] / 2), int(shave[1] / 2))).permute(2, 0,
                                                                                                           1).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, c, *padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()

        # TODO: [[a0, b0], [a1, b1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
        for i in range(x_range):
            res = self.forward(*[s[:, 0, ...] for s in split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                                                  ...], [1, 1], dim=1)], **kwargs)
            y_w_cut_unfold.append(res)
            # y_w_cut_unfold.append((s.cpu() for s in self.forward(*[s[:, 0, ...]for s in split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
            #                                    ...], [1, 1], dim=1)], **kwargs)))  # P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        if isinstance(y_w_cut_unfold[0], tuple):
            y_w_cut_unfold_out = [None] * len(y_w_cut_unfold[0])
            for i, s in enumerate(zip(*y_w_cut_unfold)):
                y_w_cut_unfold_out[i] = s
            y_w_cut_unfold = y_w_cut_unfold_out
        else:
            y_w_cut_unfold = [y_w_cut_unfold]

        y_w_cut = []
        for s_w_cut_unfold in y_w_cut_unfold:
            s_w_cut_unfold = torch.cat(s_w_cut_unfold, dim=0).cpu()
            # s_w_cut_unfold = s_w_cut_unfold.reshape(-1, c, *s_w_cut_unfold.size()[1:])
            # 109,3,30,120
            # out_size=(786, 120), k=(30, 120)
            s_w_cut = F.fold(
                s_w_cut_unfold.view(s_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                ((h - h_cut) * scale, padsize[1] * scale), (padsize[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
            s_w_cut_unfold = s_w_cut_unfold[...,
                             int(shave[0] / 2 * scale):padsize[0] * scale - int(shave[0] / 2 * scale),
                             :].contiguous()
            # 109, 3, 16, 120
            # out_size=(771, 120), k=(15, 120)
            s_w_cut_inter = F.fold(
                s_w_cut_unfold.view(s_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                ((h - h_cut - shave[0]) * scale, padsize[1] * scale),
                (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))

            s_ones = torch.ones(s_w_cut_inter.shape, dtype=s_w_cut_inter.dtype)
            divisor = F.fold(
                F.unfold(s_ones, (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
                         stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale))),
                ((h - h_cut - shave[0]) * scale, padsize[1] * scale),
                (padsize[0] * scale - shave[0] * scale, padsize[1] * scale),
                stride=(int(shave[0] / 2 * scale), int(shave[1] / 2 * scale)))
            s_w_cut_inter = s_w_cut_inter.cpu() / divisor

            s_w_cut[..., int(shave[0] / 2 * scale):(h - h_cut) * scale - int(shave[0] / 2 * scale), :] = s_w_cut_inter
            y_w_cut.append(s_w_cut)

        return y_w_cut