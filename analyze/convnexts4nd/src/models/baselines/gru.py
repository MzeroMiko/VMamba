"""Wrapper around nn.GRU to make it compatible with our RNN interface. Similar to lstm.TorchLSTM."""

import torch
from torch import nn
from src.models.sequence import SequenceModule, TransposedModule
from einops import rearrange
import src.models.nn.utils as U

@TransposedModule
class TorchGRU(nn.GRU, SequenceModule):
    """ Wrapper around nn.GRU to make it compatible with our RNN interface """

    def __init__(self, d_model, d_hidden, n_layers=1, learn_h0=False, **kwargs):
        # Rename input_size, hidden_size to d_input, d_model
        # Set batch_first as default as per this codebase's convention
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.learn_h0 = learn_h0
        super().__init__(d_model, d_hidden, num_layers=n_layers, batch_first=True, **kwargs)

        self.num_directions = 2 if self.bidirectional else 1

        if self.learn_h0:
            self.h0 = nn.Parameter(torch.zeros(self.num_layers * self.num_directions, 1, self.hidden_size))

    def step(self, x, state):
        raise NotImplementedError

    def default_state(self, *batch_shape, device=None):
        """
        Snippet from nn.LSTM source
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
        """
        if not self.learn_h0:
            h_zeros = torch.zeros(self.num_layers * self.num_directions,
                                *batch_shape, self.hidden_size,
                                dtype=torch.float, device=device)
        else:
            h_zeros = self.h0.expand(self.num_layers * self.num_directions, *batch_shape, self.hidden_size)

        return h_zeros

    @property
    def d_state(self):
        return self.n_layers * self.d_hidden

    @property
    def d_output(self):
        return self.d_hidden

    @property
    def state_to_tensor(self):
        if self.n_layers == 1:
            return lambda state: state[0]
        else:
            return lambda state: rearrange(state[0], 'd b h -> b (d h)')
