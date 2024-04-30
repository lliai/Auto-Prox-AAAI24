# https://github.com/alibaba/lightweight-neural-architecture-search/blob/main/nas/scores/compute_entropy.py MAE_DET

# Copyright (c) 2021-2022 Alibaba Group Holding Limited.

import logging
import os
import sys
import time
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import measure


def network_weight_gaussian_init(net: nn.Module, std=1):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=std)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.track_running_stats = True
                m.eps = 1e-5
                m.momentum = 1.0
                m.train()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue

    return net


@measure('entropy', bn=True)
def compute_entropy_score(
    net,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    repeat=1,
    mixup_gamma=1e-2,
    fp16=False,
):
    nas_score_list = []

    device = inputs.device
    dtype = torch.half if fp16 else torch.float32

    with torch.no_grad():
        for _ in range(repeat):
            network_weight_gaussian_init(net)
            input = torch.randn(size=list(inputs.shape),
                                device=device,
                                dtype=dtype)

            # outputs, logits = net.forward_with_features(input)
            #
            # for output in outputs:
            #     nas_score = torch.log(output.std()) + torch.log(
            #         torch.var(output / (output.std() + 1e-9)))
            #     nas_score_list.append(float(nas_score))

            # outputs = net.forward_features(input)
            net(input)
            # outputs =[]
            # for idx in [0, 6, 11]:
            #     outputs.append(net.features[idx])
            outputs = net.features


            for output in outputs:
                nas_score = torch.log(output.std()) + torch.log(
                    torch.var(output / (output.std() + 1e-9)))
                nas_score_list.append(float(nas_score))

    avg_nas_score = float(np.mean(nas_score_list))

    return avg_nas_score
