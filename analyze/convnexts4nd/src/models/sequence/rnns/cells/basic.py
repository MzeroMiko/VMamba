"""Baseline simple RNN cells such as the vanilla RNN and GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nn import LinearActivation, Activation # , get_initializer
from src.models.nn.gate import Gate
from src.models.nn.orthogonal import OrthogonalLinear
from src.models.sequence.base import SequenceModule

class CellBase(SequenceModule):
    """Abstract class for our recurrent cell interface.

    Passes input through.
    """
    registry = {}

    # https://www.python.org/dev/peps/pep-0487/#subclass-registration
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register classes with @name attribute
        if hasattr(cls, 'name') and cls.name is not None:
            cls.registry[cls.name] = cls

    name = 'id'
    valid_keys = []

    @property
    def default_initializers(self):
        return {}

    @property
    def default_architecture(self):
        return {}

    def __init__(self, d_input, d_model, initializers=None, architecture=None):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model

        self.architecture = self.default_architecture
        self.initializers = self.default_initializers
        if initializers is not None:
            self.initializers.update(initializers)
            print("Initializers:", initializers)
        if architecture is not None:
            self.architecture.update(architecture)

        assert set(self.initializers.keys()).issubset(self.valid_keys)
        assert set(self.architecture.keys()).issubset(self.valid_keys)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, input, hidden):
        """Returns output, next_state."""
        return input, input

    def default_state(self, *batch_shape, device=None):
        return torch.zeros(
            *batch_shape, self.d_model,
            device=device,
            requires_grad=False,
        )

    def step(self, x, state):
        return self.forward(x, state)

    @property
    def state_to_tensor(self):
        return lambda state: state

    @property
    def d_state(self):
        return self.d_model

    @property
    def d_output(self):
        return self.d_model


class RNNCell(CellBase):
    name = 'rnn'

    valid_keys = ['hx', 'hh', 'bias']

    default_initializers = {
        'hx': 'xavier',
        'hh': 'xavier',
    }

    default_architecture = {
        'bias': True,
    }

    def __init__(
            self, d_input, d_model,
            hidden_activation='tanh',
            orthogonal=False,
            ortho_args=None,
            zero_bias_init=False,
            **kwargs
        ):

        self.hidden_activation = hidden_activation
        self.orthogonal = orthogonal
        self.ortho_args = ortho_args
        self.zero_bias_init=zero_bias_init

        super().__init__(d_input, d_model, **kwargs)

    def reset_parameters(self):
        self.W_hx = LinearActivation(
            self.d_input, self.d_model,
            bias=self.architecture['bias'],
            zero_bias_init=self.zero_bias_init,
            initializer=self.initializers['hx'],
            activation=self.hidden_activation,
            # apply_activation=False,
            activate=False,
        )
        self.activate = Activation(self.hidden_activation, self.d_model)

        self.reset_hidden_to_hidden()

    def reset_hidden_to_hidden(self):
        if self.orthogonal:

            if self.ortho_args is None:
                self.ortho_args = {}
            self.ortho_args['d_input'] = self.d_model
            self.ortho_args['d_output'] = self.d_model

            self.W_hh = OrthogonalLinear(**self.ortho_args)
        else:
            self.W_hh = LinearActivation(
                self.d_model, self.d_model,
                bias=self.architecture['bias'],
                zero_bias_init=self.zero_bias_init,
                initializer=self.initializers['hh'],
                activation=self.hidden_activation,
                # apply_activation=False,
                activate=False,
            )
            # self.W_hh = nn.Linear(self.d_model, self.d_model, bias=self.architecture['bias'])
            # get_initializer(self.initializers['hh'], self.hidden_activation)(self.W_hh.weight)

    def forward(self, input, h):
        # Update hidden state
        hidden_preact = self.W_hx(input) + self.W_hh(h)
        hidden = self.activate(hidden_preact)

        return hidden, hidden

class GatedRNNCell(RNNCell):
    name = 'gru'

    def __init__(
            self, d_input, d_model,
            gate='G', # 'N' | 'G' | 'R' | 'UR'
            reset='G',
            **kwargs
        ):
        self.gate  = gate
        self.reset = reset
        super().__init__(d_input, d_model, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        # self.reset_gate()

    # def reset_gate(self):
        preact_ctor = LinearActivation
        preact_args = [self.d_input + self.d_model, self.d_model, self.architecture['bias']]
        self.W_g     = Gate(self.d_model, preact_ctor, preact_args, mechanism=self.gate)
        self.W_reset = Gate(self.d_model, preact_ctor, preact_args, mechanism=self.reset)

    def forward(self, input, h):
        hx = torch.cat((input, h), dim=-1)
        reset = self.W_reset(hx)

        _, update = super().forward(input, reset*h)

        g = self.W_g(hx)
        h = (1.-g) * h + g * update

        return h, h

class ExpRNNCell(RNNCell):
    """Implementation of expRNN.

    Note: there is a subtle distinction between this and the ExpRNN original cell
    in the initialization of hx, but this shouldn't make a difference.

    (Original ExpRNN cell is located in models.nn.exprnn.orthogonal.OrthogonalRNN.)
    """

    name = 'exprnn'

    def __init__(self, d_input, d_model, orthogonal=True, hidden_activation='modrelu', **kwargs):
        super().__init__(d_input, d_model, orthogonal=orthogonal, hidden_activation=hidden_activation, **kwargs)
