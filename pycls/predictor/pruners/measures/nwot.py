# Copyright 2021 Samsung Electronics Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
This contains implementations of nwot based on the updated version of
https://github.com/BayesWatch/nas-without-training
to reflect the second version of the paper https://arxiv.org/abs/2006.04647
"""

import numpy as np
import torch
from torch import nn
from . import measure
import torch.nn.functional as F

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net



@measure('nwot', bn=False)
def compute_nwot(net, inputs, targets, split_data=1, loss_fn=None):

    batch_size = len(targets)
    network_weight_gaussian_init(net)
    def counting_forward_hook(module, inp, out):
        inp = inp[0].view(inp[0].size(0), -1)
        x = (inp > 0).float()  # binary indicator
        K = x @ x.t()
        K2 = (1.0 - x) @ (1.0 - x.t())
        net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    net.K = np.zeros((batch_size, batch_size))
    for name, module in net.named_modules():
        module_type = str(type(module))
        if ('ReLU' in module_type):
            # module.register_backward_hook(counting_backward_hook)
            module.register_forward_hook(counting_forward_hook)
        if isinstance(module, torch.nn.GELU):
        # if isinstance(module, F.gelu):
        #     print('yes')
            module.visited_backwards = True
            module.register_forward_hook(counting_forward_hook)
            module.register_backward_hook(counting_backward_hook)
        # module_type = str(type(module))
        # if ('ReLU' in module_type):
        #     # module.register_backward_hook(counting_backward_hook)
        #     module.register_forward_hook(counting_forward_hook)

    x = torch.clone(inputs)
    net(x)
    s, jc = np.linalg.slogdet(net.K)
    return jc
