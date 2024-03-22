"""Wrapper around nn.LSTM to make it compatible with our RNN interface."""

import torch
from torch import nn
from src.models.sequence import SequenceModule, TransposedModule
from einops import rearrange
import src.models.nn.utils as U

@TransposedModule
class TorchLSTM(nn.LSTM, SequenceModule):
    """ Wrapper around nn.LSTM to make it compatible with our RNN interface """

    def __init__(self, d_model, d_hidden, n_layers=1, learn_h0=False, **kwargs):
        # Rename input_size, hidden_size to d_input, d_model
        # Set batch_first as default as per this codebase's convention
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.learn_h0 = learn_h0
        super().__init__(d_model, d_hidden, num_layers=n_layers, batch_first=True, **kwargs)

        self.num_directions = 2 if self.bidirectional else 1
        self.real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size

        if learn_h0:
            self.h0 = nn.Parameter(torch.zeros(self.num_layers * self.num_directions, 1, self.real_hidden_size))
            self.c0 = nn.Parameter(torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size))

    # def forward(self, x, state=None, **kwargs):
    #     # Note that state is (h_n, c_n)
    #     y = super().forward(x, state)
    #     return y, state

    def step(self, x, state):
        raise NotImplementedError("Needs to be implemented.")

    def default_state(self, *batch_shape, device=None):
        """
        Snippet from nn.LSTM source
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
        """
        if not self.learn_h0:
            h_zeros = torch.zeros(self.num_layers * self.num_directions,
                                *batch_shape, self.real_hidden_size,
                                dtype=torch.float, device=device)
            c_zeros = torch.zeros(self.num_layers * self.num_directions,
                                *batch_shape, self.hidden_size,
                                dtype=torch.float, device=device)
        else:
            h_zeros = self.h0.expand(self.num_layers * self.num_directions, *batch_shape, self.real_hidden_size)
            c_zeros = self.c0.expand(self.num_layers * self.num_directions, *batch_shape, self.hidden_size)
        return (h_zeros, c_zeros)

    @property
    def d_state(self):
        return self.n_layers * self.d_model

    @property
    def d_output(self):
        return self.d_hidden

    @property
    def state_to_tensor(self):
        if self.n_layers == 1:
            return lambda state: state[0]
        else:
            return lambda state: rearrange(state[0], 'd b h -> b (d h)')
