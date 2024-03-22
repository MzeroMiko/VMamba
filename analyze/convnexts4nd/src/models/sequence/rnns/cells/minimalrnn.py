"""Implementation of the 'MinimalRNN', which a reviewer from NeurIPS2020 asked us to compare against.

https://arxiv.org/abs/1711.06788

[21-10-22] I believe this has not been tested in awhile but should work with minimal modifications
"""

from src.models.sequence.rnns.cells.basic import CellBase
from src.models.nn import LinearActivation
from src.models.nn.gate import Gate

class MinimalRNNCell(CellBase):
    name = 'mrnn'

    valid_keys = ['hx', 'bias']

    @property
    def default_initializers(self):
        return {
            'hx': 'xavier',
        }

    @property
    def default_architecture(self):
        return {
            'bias': True,
        }


    def __init__(
            self, d_input, d_model,
            hidden_activation='tanh',
            zero_bias_init=False,
            **kwargs
        ):

        self.hidden_activation = hidden_activation
        self.zero_bias_init=zero_bias_init

        super().__init__(d_input, d_model, **kwargs,)

    def reset_parameters(self):
        self.W_hx = LinearActivation(
            self.d_input, self.d_model,
            bias=self.architecture['bias'], zero_bias_init=self.zero_bias_init,
            initializer=self.initializers['hx'], activation=self.hidden_activation,
            activate=True,
        )
        # get_initializer(self.initializers['hx'], self.hidden_activation)(self.W_hx.weight)
        # self.hidden_activation_fn = Activate(self.hidden_activation, self.d_model)

        preact_ctor = LinearActivation
        preact_args = [self.d_input + self.d_model, self.d_model, self.architecture['bias']]
        self.W_g  = Gate(self.d_model, preact_ctor, preact_args, mechanism='G')


    def forward(self, input, h):
        # Update hidden state
        # hidden_preact = self.W_hx(input)
        # hidden = self.hidden_activation_fn(hidden_preact)
        hidden = self.W_hx(input)
        hx = torch.cat((input, h), dim=-1)
        g = self.W_g(hx)
        h = (1.-g) * h + g * hidden

        return h, h
