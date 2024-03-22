"""Different deep backbone that is essentially a 1-D UNet instead of ResNet/Transformer backbone.

Sequence length gets downsampled through the depth of the network while number of feature increases.
Then sequence length gets upsampled again (causally) and blocks are connected through skip connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from einops import rearrange, repeat, reduce

import src.utils as utils
from src.models.sequence.base import SequenceModule
from src.models.sequence.modules.pool import DownPool, UpPool, up_registry, registry as down_registry
from src.models.sequence.backbones.block import SequenceResidualBlock

contract = torch.einsum

class SequenceUNet(SequenceModule):
    """UNet backbone for 1-D sequence models.

    `layer` is a Namespace that specifies '_name_', referring to a constructor, and a list of arguments to that layer constructor.
    This layer must subscribe to the interface (i) takes a hidden dimension H and sequence length L (ii) forward pass transforms input sequence of shape (B, H, L) to output (B, H, L)
    """

    def __init__(
        self,
        d_model,
        n_layers,
        pool=[],
        pool_mode='linear',
        expand=1,
        ff=2,
        cff=0,
        prenorm=False,
        dropout=0.0,
        dropres=0.0,
        layer=None,
        center_layer=None,
        residual=None,
        norm=None,
        initializer=None,
        # l_max=-1,
        transposed=True,
        # act_pool=None,
    ):
        super().__init__()
        self.d_model = d_model
        H = d_model
        self.transposed = transposed

        # Layer arguments
        layer_cfg = layer.copy()
        layer_cfg['dropout'] = dropout
        layer_cfg['transposed'] = self.transposed
        layer_cfg['initializer'] = initializer
        print("layer config", layer_cfg)

        center_layer_cfg = center_layer if center_layer is not None else layer_cfg.copy()
        center_layer_cfg['dropout'] = dropout
        center_layer_cfg['transposed'] = self.transposed

        ff_cfg = {
            '_name_': 'ffn',
            'expand': ff,
            'transposed': self.transposed,
            'activation': 'gelu',
            'initializer': initializer,
            'dropout': dropout,
        }

        def _residual(d, i, layer):
            return SequenceResidualBlock(
                d,
                i,
                prenorm=prenorm,
                dropout=dropres,
                transposed=self.transposed,
                layer=layer,
                residual=residual if residual is not None else 'R',
                norm=norm,
                pool=None,
            )

        # Down blocks
        d_layers = []
        for p in pool:
            for i in range(n_layers):
                d_layers.append(_residual(H, i+1, layer_cfg))
                if ff > 0: d_layers.append(_residual(H, i+1, ff_cfg))

            # Add sequence downsampling and feature expanding
            d_pool = utils.instantiate(down_registry, pool_mode, H, stride=p, expand=expand, transposed=self.transposed)
            d_layers.append(d_pool)
            H *= expand
        self.d_layers = nn.ModuleList(d_layers)

        # Center block
        c_layers = [ ]
        for i in range(n_layers):
            c_layers.append(_residual(H, i+1, center_layer_cfg))
            if cff > 0: c_layers.append(_residual(H, i+1, ff_cfg))
        self.c_layers = nn.ModuleList(c_layers)

        # Up blocks
        u_layers = []
        for p in pool[::-1]:
            H //= expand
            u_pool = utils.instantiate(up_registry, pool_mode, H*expand, stride=p, expand=expand, causal=True, transposed=self.transposed)
            u_layers.append(u_pool)

            for i in range(n_layers):
                u_layers.append(_residual(H, i+1, layer_cfg))
                if ff > 0: u_layers.append(_residual(H, i+1, ff_cfg))
        self.u_layers = nn.ModuleList(u_layers)

        assert H == d_model

        self.norm = nn.LayerNorm(H)

    @property
    def d_output(self):
        return self.d_model

    def forward(self, x, state=None, **kwargs):
        """
        input: (batch, length, d_input)
        output: (batch, length, d_output)
        """
        if self.transposed: x = x.transpose(1, 2)

        # Down blocks
        outputs = [] # Store all layers for SequenceUNet structure
        for layer in self.d_layers:
            outputs.append(x)
            x, _ = layer(x)

        # Center block
        outputs.append(x)
        for layer in self.c_layers:
            x, _ = layer(x)
        x = x + outputs.pop()

        for layer in self.u_layers:
            x, _ = layer(x)
            x = x + outputs.pop()

        # feature projection
        if self.transposed: x = x.transpose(1, 2) # (batch, length, expand)
        x = self.norm(x)

        return x, None # required to return a state

    def default_state(self, *args, **kwargs):
        """ x: (batch) """
        layers = list(self.d_layers) + list(self.c_layers) + list(self.u_layers)
        return [layer.default_state(*args, **kwargs) for layer in layers]

    def step(self, x, state, **kwargs):
        """
        input: (batch, d_input)
        output: (batch, d_output)
        """
        # States will be popped in reverse order for convenience
        state = state[::-1]

        # Down blocks
        outputs = [] # Store all layers for SequenceUNet structure
        next_state = []
        for layer in self.d_layers:
            outputs.append(x)
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            if x is None: break

        # Center block
        if x is None:
            # Skip computations since we've downsized
            skipped = len(self.d_layers) - len(outputs)
            for _ in range(skipped+len(self.c_layers)+skipped):
                next_state.append(state.pop())
            u_layers = list(self.u_layers)[skipped:]
        else:
            outputs.append(x)
            for layer in self.c_layers:
                x, _next_state = layer.step(x, state=state.pop(), **kwargs)
                next_state.append(_next_state)
            x = x + outputs.pop()
            u_layers = self.u_layers

        for layer in u_layers:
            x, _next_state = layer.step(x, state=state.pop(), **kwargs)
            next_state.append(_next_state)
            x = x + outputs.pop()

        # feature projection
        x = self.norm(x)
        return x, next_state
