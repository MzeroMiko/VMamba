"""Wrapper around nn.Conv1d to adhere to SequenceModule interface."""

import torch
import torch.nn.functional as F
from torch import nn
import hydra
from models.sequence.base import SequenceModule
from einops import rearrange

import src.models.nn.utils as U
from src.models.nn import Activation

class Conv1d(SequenceModule):
    """ Simple wrapper for nn.Conv1d """
    def __init__(self, d_model, *args, d_output=None, activation='gelu', dropout=0.0, transposed=True, **kwargs):
        # Accepted kwargs passed into Conv1d interface
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        super().__init__()

        self.d_model = d_model
        if d_output is None: d_output = d_model
        self.d_output = d_output
        self.transposed = transposed
        self.conv1d = nn.Conv1d(d_model, d_output, *args, **kwargs)
        self.activation = Activation(activation)

    def forward(self, x, resolution=None, state=None, *args, **kwargs):
        if not self.transposed: x = x.transpose(-1, -2)
        y = self.conv1d(x)
        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)
        return y, None

    def step(self, x, state):
        raise NotImplementedError
