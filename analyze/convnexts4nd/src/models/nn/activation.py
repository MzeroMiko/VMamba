"""Utilities for activation functions."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def Activation(activation=None, size=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear', 'none' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation.startswith('glu-'):
        return GLU(dim=dim, activation=activation[4:])
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softplus':
        return nn.Softplus()
    elif activation == 'modrelu':
        return ModReLU(size)
    elif activation in ['sqrelu', 'relu2']:
        return SquaredReLU()
    elif activation == 'laplace':
        return Laplace()
    # Earlier experimentation with a LN in the middle of the block instead of activation
    # IIRC ConvNext does something like this?
    # elif activation == 'ln':
    #     return TransposedLN(dim)
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

class GLU(nn.Module):
    def __init__(self, dim=-1, activation='sigmoid'):
        super().__init__()
        assert not activation.startswith('glu')
        self.dim = dim
        self.activation_fn = Activation(activation)

    def forward(self, x):
        x, g = torch.split(x, x.size(self.dim)//2, dim=self.dim)
        return x * self.activation_fn(g)

class ModReLU(nn.Module):
    # Adapted from https://github.com/Lezcano/expRNN

    def __init__(self, features):
        # For now we just support square layers
        super().__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = F.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude


class SquaredReLU(nn.Module):
    def forward(self, x):
        # return F.relu(x)**2
        return torch.square(F.relu(x))  # Could this be faster?

def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))

class Laplace(nn.Module):
    def __init__(self, mu=0.707107, sigma=0.282095):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return laplace(x, mu=self.mu, sigma=self.sigma)


