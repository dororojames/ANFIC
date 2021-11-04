from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def gen_condition(lmdas, batch_size, shuffle=False, device='cpu'):
    if not isinstance(lmdas, list) and not isinstance(lmdas, tuple):
        lmdas = [lmdas]
    lmdas = lmdas * int(np.ceil(batch_size/len(lmdas)))
    if shuffle:
        np.random.shuffle(lmdas)
    return torch.Tensor(lmdas[:batch_size]).view(-1, 1).to(device=device)


def gen_discrete_condition(lmdas, batch_size, shuffle=False, device='cpu'):
    if not isinstance(lmdas, list) and not isinstance(lmdas, tuple):
        lmdas = [lmdas]
    lmda_map = dict(zip(lmdas, torch.eye(len(lmdas)).unbind(0)))
    lmdas = lmdas * int(np.ceil(batch_size/len(lmdas)))
    if shuffle:
        np.random.shuffle(lmdas)
    conds = []
    for lmda in lmdas:
        conds.append(lmda_map[lmda])
    return torch.stack(conds).to(device=device), torch.Tensor(lmdas[:batch_size]).view(-1, 1).to(device=device)


def gen_random_condition(lmdas, batch_size, shuffle=False, device='cpu'):
    if not isinstance(lmdas, list) and not isinstance(lmdas, tuple):
        lmdas = [lmdas]
    rands = np.exp(np.random.uniform(np.log(np.min(lmdas)), np.log(
        np.max(lmdas)), batch_size-len(lmdas)))
    lmdas = np.concatenate([np.array(lmdas), rands])
    if shuffle:
        np.random.shuffle(lmdas)
    else:
        lmdas = np.sort(lmdas)
    return torch.Tensor(lmdas[:batch_size]).view(-1, 1).to(device=device)


def hasout_channels(module: nn.Module):
    return hasattr(module, 'out_channels') or hasattr(module, 'out_features') or hasattr(module, 'num_features') or hasattr(module, 'hidden_size')


def get_out_channels(module: nn.Module):
    if hasattr(module, 'out_channels'):
        return module.out_channels
    elif hasattr(module, 'out_features'):
        return module.out_features
    elif hasattr(module, 'num_features'):
        return module.num_features
    elif hasattr(module, 'hidden_size'):
        return module.hidden_size
    raise AttributeError(
        str(module)+" has no avaiable output channels attribute")


class ConditionalLayer(nn.Module):

    def __init__(self, module: nn.Module, out_channels=None, discrete=False, conditions: int = 1, ver=1):
        super(ConditionalLayer, self).__init__()
        self.m = module
        self.discrete = discrete
        assert conditions >= 0, conditions
        self.condition_size = conditions
        self.ver = ver
        if conditions:
            if out_channels is None:
                out_channels = get_out_channels(module)
            self.out_channels = out_channels

            if self.ver == 1:
                self.weight = nn.Parameter(
                    torch.Tensor(conditions, out_channels*2))
                nn.init.kaiming_normal_(self.weight)
            else:
                self.affine = nn.Sequential(
                    nn.Linear(conditions, 16),
                    nn.Sigmoid(),
                    nn.Linear(16, out_channels*2, bias=False)
                )

    def extra_repr(self):
        if self.ver == 1:
            s = '(condition): '
            if self.condition_size:
                s += 'Condition({condition_size}, {out_channels})'
            else:
                s += 'skip'
            return s.format(**self.__dict__)
        else:
            return ""

    def _set_condition(self, condition):
        self.condition = condition

    def forward(self, *input, condition=None):
        output = self.m(*input)

        if self.condition_size:
            # print('cond')
            if condition is None:
                condition = self.condition

            if not isinstance(condition, tuple):
                BC, BO = condition.size(0), output.size(
                    0)  # legacy problem for multi device
                if BC != BO:
                    assert BC % BO == 0 and output.is_cuda, "{}, {}, {}".format(
                        condition.size(), output.size(), output.device)
                    idx = int(str(output.device)[-1])
                    condition = condition[BO*idx:BO*(idx+1)]
                    # print(idx, condition.cpu().numpy())
                if condition.device != output.device:
                    condition = condition.to(output.device)

                if self.ver == 1:
                    condition = condition.mm(self.weight)
                else:
                    condition = self.affine(condition)

                scale, bias = condition.view(
                    condition.size(0), -1, *(1,)*(output.dim()-2)).chunk(2, dim=1)
                self.condition = (scale, bias)
            else:
                # print("reuse")
                scale, bias = condition

            output = output * F.softplus(scale) + bias

        return output.contiguous()


def conditional_warping(m: nn.Module, types=(nn.modules.conv._ConvNd), **kwargs):
    def dfs(sub_m: nn.Module, prefix=""):
        for n, chd_m in sub_m.named_children():
            if dfs(chd_m, prefix+"."+n if prefix else n):
                setattr(sub_m, n, ConditionalLayer(chd_m, **kwargs))
        else:
            if isinstance(sub_m, types):
                # print(prefix, "C")
                return True
            else:
                pass
                # print(prefix)
        return False

    dfs(m)
    print(m)


def set_condition(model, condition):
    for m in model.modules():
        if isinstance(m, ConditionalLayer):
            m._set_condition(condition)


def _args_expand(*args, length):
    for arg in args:
        yield [arg] * length


class ConditionalSequantial(nn.Sequential):

    def __init__(self, *args, out_channels=None, discrete=False, conditions=1, conv_only=True):
        out_channels, discrete, conditions = _args_expand(
            out_channels, discrete, conditions, length=len(args))
        super(ConditionalSequantial, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            _dict = args[0]
        else:
            _dict = OrderedDict(zip(map(str, range(len(args))), args))

        for idx, (key, module) in enumerate(_dict.items()):
            if conv_only and not isinstance(module, nn.modules.conv._ConvNd):
                conditions[idx] = 0
            if not hasout_channels(module) and out_channels[idx] is None and idx:
                out_channels[idx] = get_out_channels(self[idx-1])
            module = ConditionalLayer(
                module, out_channels=out_channels[idx], discrete=discrete[idx], conditions=conditions[idx])
            self.add_module(key, module)

    def _set_condition(self, condition):
        for module in self:
            module._set_condition(condition)

    def forward(self, input, condition=None):
        for module in self:
            input = module(input, condition=condition)
        return input
