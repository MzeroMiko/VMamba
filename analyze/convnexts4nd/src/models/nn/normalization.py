"""Normalization modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Normalization(nn.Module):
    def __init__(
        self,
        d,
        transposed=False, # Length dimension is -1 or -2
        _name_='layer',
        **kwargs
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == 'layer':
            self.channel = True # Normalize over channel dimension
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == 'instance':
            self.channel = False
            norm_args = {'affine': False, 'track_running_stats': False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(d, **norm_args) # (True, True) performs very poorly
        elif _name_ == 'batch':
            self.channel = False
            norm_args = {'affine': True, 'track_running_stats': True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == 'group':
            self.channel = False
            self.norm = nn.GroupNorm(1, d, **kwargs)
        elif _name_ == 'none':
            self.channel = True
            self.norm = nn.Identity()
        else: raise NotImplementedError

    def forward(self, x):
        # Handle higher dimension logic
        shape = x.shape
        if self.transposed:
            x = rearrange(x, 'b d ... -> b d (...)')
        else:
            x = rearrange(x, 'b ... d -> b (...) d')

        # The cases of LayerNorm / no normalization are automatically handled in all cases
        # Instance/Batch Norm work automatically with transposed axes
        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x

    def step(self, x, **kwargs):
        assert self._name_ in ["layer", "none"]
        if self.transposed: x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed: x = x.squeeze(-1)
        return x

class TransposedLN(nn.Module):
    """LayerNorm module over second dimension.

    Assumes shape (B, D, L), where L can be 1 or more axis.
    This is slow and a dedicated CUDA/Triton implementation shuld provide substantial end-to-end speedup.
    """
    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            # calc. stats over D dim / channels
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            # move channel to last axis, apply layer_norm, then move channel back to second axis
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y

class TSNormalization(nn.Module):

    def __init__(self, method, horizon):
        super().__init__()

        self.method = method
        self.horizon = horizon


    def forward(self, x):
        # x must be BLD
        if self.method == 'mean':
            self.scale = x.abs()[:, :-self.horizon].mean(dim=1)[:, None, :]
            return x / self.scale
        elif self.method == 'last':
            self.scale = x.abs()[:, -self.horizon-1][:, None, :]
            return x / self.scale
        return x

class TSInverseNormalization(nn.Module):

    def __init__(self, method, normalizer):
        super().__init__()

        self.method = method
        self.normalizer = normalizer

    def forward(self, x):
        if self.method == 'mean' or self.method == 'last':
            return x * self.normalizer.scale
        return x

class ReversibleInstanceNorm1dInput(nn.Module):
    def __init__(self, d, transposed=False):
        super().__init__()
        # BLD if transpoed is False, otherwise BDL
        self.transposed = transposed
        self.norm = nn.InstanceNorm1d(d, affine=True, track_running_stats=False)

    def forward(self, x):
        # Means, stds
        if not self.transposed:
            x = x.transpose(-1, -2)

        self.s, self.m = torch.std_mean(x, dim=-1, unbiased=False, keepdim=True)
        self.s += 1e-4

        x = (x - self.m) / self.s
        # x = self.norm.weight.unsqueeze(-1) * x + self.norm.bias.unsqueeze(-1)

        if not self.transposed:
            return x.transpose(-1, -2)
        return x

class ReversibleInstanceNorm1dOutput(nn.Module):

    def __init__(self, norm_input):
        super().__init__()
        self.transposed = norm_input.transposed
        self.weight = norm_input.norm.weight
        self.bias = norm_input.norm.bias
        self.norm_input = norm_input

    def forward(self, x):
        if not self.transposed:
            x = x.transpose(-1, -2)

        # x = (x - self.bias.unsqueeze(-1))/self.weight.unsqueeze(-1)
        x = x * self.norm_input.s + self.norm_input.m

        if not self.transposed:
            return x.transpose(-1, -2)
        return x
