"""Wrapper around nn.Conv2d to adhere to SequenceModule interface."""

import torch
from torch import nn

from src.models.sequence.base import SequenceModule
from src.models.nn import Activation, DropoutNd

class Conv2d(SequenceModule):
    """ Simple wrapper for nn.Conv1d """
    def __init__(self, d_model, d_output=None, activation='gelu', depthwise=False, dropout=0.0, tie_dropout=False, transposed=True, **kwargs):
        # kwargs passed into Conv2d interface:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        super().__init__()

        valid_kwargs = ["in_channels","out_channels","kernel_size","stride",
                "padding","padding_mode","dilation","groups","bias"]
        kwargs = {k:v for k,v in kwargs.items() if k in valid_kwargs}

        self.d_model = d_model
        if d_output is None: d_output = d_model
        self.d_output = d_output
        self.transposed = transposed
        self.depthwise = depthwise

        if self.depthwise:
            self.conv2d = nn.Conv2d(d_model, d_model, padding='same', groups=d_model, **kwargs)
            self.linear = nn.Conv2d(d_model, d_output, 1, 1)
        else:
            self.conv2d = nn.Conv2d(d_model, d_output, padding='same', **kwargs)
            self.linear = nn.Identity()
            dropout_fn = DropoutNd if tie_dropout else nn.Dropout
            self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = Activation(activation)

    def forward(self, x, resolution=None, state=None, *args, **kwargs):
        if not self.transposed: x = x.transpose(-1, -3)
        y = self.conv2d(x)
        y = self.activation(y) # NOTE doesn't work with glu
        y = self.dropout(y)
        y = self.linear(y)
        if not self.transposed: y = y.transpose(-1, -3)
        return y, None

    def step(self, x, state):
        raise NotImplementedError
