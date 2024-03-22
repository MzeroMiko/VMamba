"""Implementation of modular block design used in S4. Compatible with other kernels."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from functools import partial
from einops import rearrange, repeat

from src.models.nn import LinearActivation, Activation, DropoutNd
from src.models.sequence.base import SequenceModule
from src.models.sequence.kernels.fftconv import FFTConv
import src.utils as utils
import src.utils.registry as registry

import src.utils.train
log = src.utils.train.get_logger(__name__)

contract = torch.einsum


class S4Block(SequenceModule):
    """General block design wrapping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    def __init__(
        self,
        d_model,
        bottleneck=None,
        activation='gelu',
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        layer='fftconv',
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    transposed=False,
                    initializer=initializer,
                    activation=None,
                    activate=False,
                    weight_norm=weight_norm,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        # self.layer = FFTConv(d_model, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)
        layer_cfg = layer_args.copy()
        layer_cfg['_name_'] = layer
        layer_cfg['transposed'] = False
        layer_cfg['dropout'] = dropout
        self.layer = utils.instantiate(registry.layer, layer_cfg, d_model)

        # Pointwise operations
        # Activation after layer
        self.activation = Activation(activation)

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.layer.d_output,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )



    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        y, state = self.layer(x, **kwargs)

        y = self.activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        y = self.activation(y)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor
