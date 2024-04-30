import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def act_forward_conv2d(self, x):
    x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                 self.dilation, self.groups)
    # intercept and store the activations after passing through
    # 'hooked' identity op
    self.act = self.dummy(x)
    return self.act


def act_forward_linear(self, x):
    x = F.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act


def act_forward_relu(self, x):
    return self.dummy(x)


class Swish(nn.Module):

    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


def convert_model_to_act(model, act='tanh'):
    if act == 'leaky':
        dummy_function = nn.LeakyReLU
    elif act == 'tanh':
        dummy_function = nn.Tanh
    elif act == 'swish':
        dummy_function = Swish
    elif act == 'sigmoid':
        dummy_function = nn.Sigmoid
    elif act == 'prelu':
        dummy_function = nn.PReLU
    elif act == 'relu':
        raise NotImplementedError
    elif act is None:
        return model

    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            layer.dummy = dummy_function()

            layer.forward = types.MethodType(act_forward_relu, layer)

            # function to call during backward pass
            # (hooked on identity op at output of layer)
            def hook_factory(layer):

                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad),
                                         list(range(2, len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del layer.act
                    # without deleting this, a nasty memory leak occurs!
                    # related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555 # noqa: E501

                return hook

            # register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))

    return model


class ScaleIdentity(nn.Identity):

    def __init__(self, alpha=1, *args, **kwargs):
        super(ScaleIdentity, self).__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        return self.alpha * input


def skip_forward_identity(self, x):
    # intercept and store the activations after passing through
    # 'hooked' identity op
    return self.dummy(x)


def convert_model_skip(model, alpha=1):
    dummy_function = ScaleIdentity
    for layer in model.modules():
        if isinstance(layer, nn.Identity):
            layer.dummy = dummy_function(alpha=alpha)

            layer.forward = types.MethodType(skip_forward_identity, layer)

            # function to call during backward pass
            # (hooked on identity op at output of layer)
            def hook_factory(layer):

                def hook(module, grad_input, grad_output):
                    act = layer.act.detach()
                    grad = grad_output[0].detach()
                    if len(act.shape) > 2:
                        g_nk = torch.sum((act * grad),
                                         list(range(2, len(act.shape))))
                    else:
                        g_nk = act * grad
                    del_k = g_nk.pow(2).mean(0).mul(0.5)
                    if layer.fisher is None:
                        layer.fisher = del_k
                    else:
                        layer.fisher += del_k
                    del layer.act
                    # without deleting this, a nasty memory leak occurs!
                    # related: https://discuss.pytorch.org/t/memory-leak-when-using-forward-hook-and-backward-hook-simultaneously/27555 # noqa: E501

                return hook

            # register backward hook on identity fcn to compute fisher info
            layer.dummy.register_backward_hook(hook_factory(layer))

    return model
