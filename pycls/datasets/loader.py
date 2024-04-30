#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Modified from pycls.
https://github.com/facebookresearch/pycls/blob/main/pycls/datasets/loader.py
"""

import os

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from pycls.core.config import cfg
from pycls.datasets.chaoyang import Chaoyang
from pycls.datasets.cifar100 import Cifar100
from pycls.datasets.flowers import Flowers
from pycls.datasets.imagenet import ImageNet



_DATASETS = {
    'cifar100': Cifar100,
    'flowers': Flowers,
    'chaoyang': Chaoyang,
    'imagenet': ImageNet,
}

_DATA_DIR = os.path.join('.', 'data')
if not os.path.exists(_DATA_DIR):
    os.makedirs(_DATA_DIR)

_PATHS = {
    'cifar100': '',
    'flowers': 'flowers',
    'chaoyang': 'chaoyang',
    # 'imagenet': 'ImageNet',
    'imagenet': 'imagenet-10p-subset'
}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    err_str = "Dataset '{}' not supported".format(dataset_name)
    assert dataset_name in _DATASETS and dataset_name in _PATHS, err_str
    data_path = os.path.join(_DATA_DIR, _PATHS[dataset_name])
    dataset = _DATASETS[dataset_name](data_path, split)
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader


def construct_train_loader():
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True,
    )


def construct_test_loader():
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )



def construct_proxy_loader():
    c100_loader = _construct_loader(
        dataset_name='cifar100',
        split='train',
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    flower_loader = _construct_loader(
        dataset_name='flowers',
        split='train',
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    chaoyang_loader = _construct_loader(
        dataset_name='chaoyang',
        split='train',
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    imagenet_loader = _construct_loader(
        dataset_name='imagenet',
        split='train',
        batch_size=128,
        shuffle=True,
        drop_last=True,
    )
    if cfg.PROXY_DATASET == 'all' and cfg.MODEL.TYPE == 'AutoFormerSub':
        return c100_loader, flower_loader, chaoyang_loader, imagenet_loader
    elif cfg.PROXY_DATASET == 'all' and cfg.MODEL.TYPE == 'PiT':
        return c100_loader, flower_loader, chaoyang_loader
    else:
        return _construct_loader(
            dataset_name=cfg.PROXY_DATASET,
            split=cfg.TRAIN.SPLIT,
            batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
            shuffle=True,
            drop_last=True,
        )





def shuffle(loader, cur_epoch):
    err_str = "Sampler type '{}' not supported".format(type(loader.sampler))
    assert isinstance(loader.sampler,
                      (RandomSampler, DistributedSampler)), err_str
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(cur_epoch)
