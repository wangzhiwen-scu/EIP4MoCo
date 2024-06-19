import os
import sys
import subprocess
import torch
from torch import nn
import torch.multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from logging import info as log_string
import warnings
import sys

sys.path.append('.') 
from layers.dftl_config import Config

Warning("No module named 'apex")

def scaled_all_reduce(tensors):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    process group.
    """
    # There is no need for reduction in the single-proc case
    gpus = dist.get_world_size()
    if gpus == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / gpus)
    return tensors

def init_dist(launcher, args, backend='nccl', **kwargs):
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')

def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    # print(f"DDP: {dist.is_available()} {world_size}")
    return rank, world_size

def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None, **kwargs):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # print(proc_id, ntasks, node_list, addr)
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    # print(os.environ)
    dist.init_process_group(backend=backend)

def reduce_mean(tensor, nprocs=None):
    if nprocs is None:
        _, nprocs = get_dist_info()
        if nprocs == 1:
            return tensor
    # print("reduce_mean", tensor)
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    # print(rt, nprocs)
    rt /= nprocs
    # print(rt)
    return rt

class MMDistributedDataParallel(DistributedDataParallel):

    def __init__(self, model, device_ids):
        super(MMDistributedDataParallel, self).__init__(model, device_ids, find_unused_parameters=True)

        self.ddp = model

    def reduce_mean(self, tensor, nprocs=None):
        if nprocs is None:
            _, nprocs = get_dist_info()
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= nprocs
        return rt

    def ddp_step(self, loss_dicts):
        losses = {}
        _, world_size = get_dist_info()
        if world_size == 1:
            return loss_dicts
        dist.barrier()
        # keys = loss_dicts.keys()
        # reduced_loss = scaled_all_reduce(loss_dicts.values())
        # losses = {k: v for k, v in zip(keys, reduced_loss)}
        for k, loss in loss_dicts.items():
            reduced_loss = self.reduce_mean(loss)
            losses.update({k: reduced_loss})
        return losses

def dist_train_v1(args, model):
    if args.mode == "DDP":
        # if args.global_rank == 0:
        #     log_string(f'Distributed training: {args.distributed}')
        # if args.distributed:
        #     if args.amp is not None:
        #         if not args.amp:
        #             # delay_allreduce delays all communication to the end of the backward pass.
        #             log_string("IN apex DistributedDataParallel mode.")
        #             model = DDP(model, delay_allreduce=True)
        #     else:
        #         # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        model = MMDistributedDataParallel(model, device_ids=[args.local_rank])
                # train_sampler = torch.auxiliary.data.distributed.DistributedSampler(train_dataset)
                # val_sampler = torch.auxiliary.data.distributed.DistributedSampler(val_dataset)
    elif args.mode == "DP":
        log_string(f'DataParallel training')
        model = nn.DataParallel(model, device_ids=args.device_ids)

    return model

# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Time    : 2022
# @Author  : Xiao Wu
# @reference:



class TaskDispatcher(Config):
    _task = dict()

    def __init_subclass__(cls, name='', **kwargs):
        super().__init_subclass__(**kwargs)

        if name != '':
            cls._task[name] = cls
            cls._name = name
            # print(cls.__repr__, cls..__repr__)
        else:
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        if cls is TaskDispatcher:
            task = kwargs.get('task')
            try:
                cls = cls._task[task]
            except KeyError:
                raise ValueError(f'Got task={task} but expected'
                                 f'one of {cls._task.keys()}')

        instance = super().__new__(cls)

        return instance

    # def __len__(self):
    #     return len(self._cfg_dict)
    #
    # def __getattr__(self, name):
    #     return getattr(self._cfg_dict, name)
    #
    # def __delattr__(self, name):
    #     return delattr(self._cfg_dict, name)
    #
    # def __getitem__(self, name):
    #     return self._cfg_dict.__getitem__(name)
    #
    # def __iter__(self):
    #     return iter(self._cfg_dict)
    #
    # def __repr__(self):
    #     return f'TaskDispatcher {self._cfg_dict.__repr__()}'

    # def __setattr__(self, name, value):
    #     if isinstance(value, dict):
    #         value = ConfigDict(value)
    #     print("__setattr__")
    #     self._cfg_dict.__setattr__(name, value)

    # def __setitem__(self, name, value):
    #     if isinstance(value, dict):
    #         value = ConfigDict(value)
    #     print("__setitem__")
    #     self._cfg_dict.__setitem__(name, value)

    @classmethod
    def new(cls, **kwargs):
        # 需要从外部启动和从任务启动，但参数不同
        key = 'mode'
        value = kwargs.setdefault('mode', None)
        print('111-mode:', value)
        if value is None:
            # 第二、三调用层进入此函数
            key = 'arch'
            if kwargs.get('task', None):
                # 二
                value = kwargs.pop('task')
                print('222-task: ', value)
            elif kwargs.get('arch', None):
                # 三
                key = 'arch'
                value = kwargs.pop('arch')
                print('333-arch:', value)
            else:
                key = 'arch'

        kwargs.pop('mode')

        try:
            cls = cls._task[value]
        except KeyError:
            warning = f'Got {key}={value} but expected ' \
                      f'one of {cls._task.keys()}'
            warnings.warn(warning)
            return Config()

        # print("kwargs: ", cls, kwargs)
        return cls(**kwargs)

class ModelDispatcher(object):
    _task = dict()

    def __init_subclass__(cls, name='', **kwargs):
        super().__init_subclass__(**kwargs)
        if name != '':
            cls._task[name] = cls
            cls._name = name
            # print(cls.__repr__, cls..__repr__)
        else:
            # warnings.warn(f'Creating a subclass of MetaModel {cls.__name__} with no name.')
            cls._task[cls.__name__] = cls
            cls._name = cls.__name__

    def __new__(cls, *args, **kwargs):
        if cls is ModelDispatcher:
            task = kwargs.get('task')
            try:
                cls = cls._task[task]
            except KeyError:
                raise ValueError(f'Got task={task} but expected'
                                 f'one of {cls._task.keys()}')

        instance = super().__new__(cls)

        return instance

    @classmethod
    def build_model(cls, cfg):

        arch = cfg.arch
        task = cfg.task
        model_style = cfg.model_style

        try:
            #获得PansharpeningModel,进行分发
            cls = cls._task[task](None, None)
        except KeyError:
            raise ValueError(f'Got task={task} but expected '
                             f'one of {cls._task.keys()} in {cls}')
        try:
            # 获得具体的模型
            cls_arch = cls._models[arch]()
        except KeyError:
            raise ValueError(f'Got arch={arch} but expected '
                             f'one of {cls._models.keys()} in {cls}')

        model, criterion, optimizer, scheduler = cls_arch(cfg)

        if model_style is None:
            # 获得PansharpeningModel,model+head
            model_style = task

        if model_style is not None:
            try:
                # 获得具体的模型
                model = cls._task[model_style](model, criterion)
            except KeyError:
                raise ValueError(f'Got model_style={model_style} but expected '
                                 f'one of {cls._models.keys()} (merged in _models) in {cls}')

        return model, criterion, optimizer, scheduler