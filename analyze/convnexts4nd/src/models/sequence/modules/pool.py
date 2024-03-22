"""Implements downsampling and upsampling on sequences."""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

from src.models.sequence import SequenceModule
from src.models.nn import LinearActivation


"""The following pooling modules all subscribe to the same interface.

stride: Subsample on the layer dimension.
expand: Expansion factor on the feature dimension.
"""
class DownSample(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        if x is None: return None
        if self.stride > 1:
            assert x.ndim == 3, "Downsampling with higher-dimensional inputs is currently not supported. It is recommended to use average or spectral pooling instead."
            if self.transposed:
                x = x[..., 0::self.stride]
            else:
                x = x[..., 0::self.stride, :]

        if self.expand > 1:
            if self.transposed:
                x = repeat(x, 'b d ... -> b (d e) ...', e=self.expand)
            else:
                x = repeat(x, 'b ... d -> b ... (d e)', e=self.expand)

        return x, None


    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand

class DownAvgPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=None, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        if self.expand is not None:
            self.linear = LinearActivation(
                d_input,
                d_input * expand,
                transposed=transposed,
            )

    def forward(self, x):
        if not self.transposed:
            x = rearrange(x, 'b ... d -> b d ...')

        if self.stride > 1:
            # einops appears slower than F
            if x.ndim == 3:
                x = F.avg_pool1d(x, self.stride, self.stride)
            elif x.ndim == 4:
                x = F.avg_pool2d(x, self.stride, self.stride)
            else:
                # Reduction string e.g. "b d (l1 2) (l2 2) -> b d l1 l2"
                reduce_str = "b d " + " ".join([f"(l{i} {self.stride})" for i in range(x.ndim-2)]) \
                        + " -> b d " + " ".join([f"l{i}" for i in range(x.ndim-2)])
                x = reduce(x, reduce_str, 'mean')

        # if self.expand > 1:
        #     x = repeat(x, 'b d ... -> b (d e) ...', e=self.expand)

        if not self.transposed:
            x = rearrange(x, 'b d ... -> b ... d')
        if self.expand is not None:
            x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        if self.expand is None:
            return self.d_input
        else:
            return self.d_input * self.expand

class DownSpectralPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        """
        x: (B, L..., D)
        """
        if not self.transposed:
            x = rearrange(x, 'b ... d -> b d ...')
        shape = x.shape[2:]
        x_f = torch.fft.ifftn(x, s=shape)

        for axis, l in enumerate(shape):
            assert l % self.stride == 0, 'input length must be divisible by stride'
            new_l = l // self.stride
            idx = torch.cat([torch.arange(0, new_l-new_l//2), l+torch.arange(-new_l//2, 0)]).to(x_f.device)
            x_f = torch.index_select(x_f, 2+axis, idx)
        x = torch.fft.ifftn(x_f, s=[l//self.stride for l in shape])
        x = x.real

        if self.expand > 1:
            x = repeat(x, 'b d ... -> b (d e) ...', e=self.expand)
        if not self.transposed:
            x = rearrange(x, 'b d ... -> b ... d')
        return x, None

    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

    @property
    def d_output(self):
        return self.d_input * self.expand

class UpSample(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, transposed=True):
        super().__init__()
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

    def forward(self, x):
        if x is None: return None
        if self.expand > 1:
            if self.transposed:
                x = reduce(x, '... (d e) l -> ... d l', 'mean', e=self.expand)
            else:
                x = reduce(x, '... (d e) -> ... d', 'mean', e=self.expand)
        if self.stride > 1:
            if self.transposed:
                x = repeat(x, '... l -> ... (l e)', e=self.stride)
            else:
                x = repeat(x, '... l d -> ... (l e) d', e=self.stride)
        return x, None

    @property
    def d_output(self):
        return self.d_input // self.expand
    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

class UpAvgPool(SequenceModule):
    def __init__(self, d_input, stride=1, expand=1, causal=False, transposed=True):
        super().__init__()
        assert d_input % expand == 0
        self.d_input = d_input
        self.stride = stride
        self.expand = expand
        self.causal = causal
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            d_input // expand,
            transposed=transposed,
        )

    def forward(self, x):
        # TODO only works for 1D right now
        if x is None: return None
        x = self.linear(x)
        if self.stride > 1:
            if self.transposed:
                if self.causal:
                    x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
                x = repeat(x, '... l -> ... (l e)', e=self.stride)
            else:
                if self.causal:
                    x = F.pad(x[..., :-1, :], (0, 0, 1, 0)) # Shift to ensure causality
                x = repeat(x, '... l d -> ... (l e) d', e=self.stride)
        return x, None

    @property
    def d_output(self):
        return self.d_input // self.expand
    def step(self, x, state, **kwargs):
        if self.stride > 1 or self.expand > 1:
            raise NotImplementedError
        return x, state

class DownLinearPool(SequenceModule):
    def __init__(self, d_model, stride=1, expand=1, causal=False, transposed=True):
        super().__init__()

        self.d_model = d_model
        self.stride = stride
        self.expand = expand
        self.transposed = transposed

        self.linear = LinearActivation(
            d_model * stride,
            d_model * expand,
            transposed=transposed,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.stride)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.stride)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        if x is None: return None, state
        state.append(x)
        if len(state) == self.stride:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *batch_shape, device=None):
        return []

    @property
    def d_output(self):
        return self.d_input * self.expand

class UpLinearPool(SequenceModule):
    def __init__(self, d, stride=1, expand=1, causal=False, transposed=True):
        super().__init__()

        assert d % expand == 0
        self.d_model = d
        self.d_output = d // expand
        # self._d_output = d_output
        self.stride = stride
        self.causal = causal
        self.transposed = transposed

        self.linear = LinearActivation(
            self.d_model,
            self.d_output * stride,
            transposed=transposed,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)
        if self.transposed:
            if self.causal:
                x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
            x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.stride)
        else:
            if self.causal:
                x = F.pad(x[..., :-1, :], (0, 0, 1, 0)) # Shift to ensure causality
            x = rearrange(x, '... l (h s) -> ... (l s) h', s=self.stride)
        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.stride)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.stride), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state


"""Pooling functions with trainable parameters."""

class DownPool2d(SequenceModule):

    def __init__(self, d_input, d_output, stride=1, transposed=True, weight_norm=True):
        # TODO make d_output expand instead
        super().__init__()

        self.linear = LinearActivation(
            d_input,
            d_output,
            transposed=transposed,
            weight_norm=weight_norm,
        )

        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride),

    def forward(self, x):
        if self.transposed:
            x = self.pool(x)

# TODO DownPool/UpPool are currently used by unet/sashimi backbones
# DownLinearPool is used by the registry (for isotropic backbone)
# DownPool is essentially the same as DownLinearPool. These should be consolidated
class DownPool(SequenceModule):
    def __init__(self, d_input, d_output=None, expand=None, stride=1, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()
        assert (d_output is None) + (expand is None) == 1
        if d_output is None: d_output = d_input * expand

        self.d_output = d_output
        self.stride = stride
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input * stride,
            d_output,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x):
        if self.transposed:
            x = rearrange(x, '... h (l s) -> ... (h s) l', s=self.stride)
        else:
            x = rearrange(x, '... (l s) h -> ... l (h s)', s=self.stride)
        x = self.linear(x)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        if x is None: return None, state
        state.append(x)
        if len(state) == self.stride:
            x = rearrange(torch.stack(state, dim=-1), '... h s -> ... (h s)')
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            return x, []
        else:
            return None, state

    def default_state(self, *batch_shape, device=None):
        return []


class UpPool(SequenceModule):
    def __init__(self, d_input, d_output, stride, transposed=True, weight_norm=True, initializer=None, activation=None):
        super().__init__()

        self.d_input = d_input
        self._d_output = d_output
        self.stride = stride
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            d_output * stride,
            transposed=transposed,
            initializer=initializer,
            weight_norm = weight_norm,
            activation=activation,
            activate=True if activation is not None else False,
        )

    def forward(self, x, skip=None):
        x = self.linear(x)
        if self.transposed:
            x = F.pad(x[..., :-1], (1, 0)) # Shift to ensure causality
            x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.stride)
        else:
            x = F.pad(x[..., :-1, :], (0, 0, 1, 0)) # Shift to ensure causality
            x = rearrange(x, '... l (h s) -> ... (l s) h', s=self.stride)
        if skip is not None:
            x = x + skip
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """

        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            if self.transposed: x = x.unsqueeze(-1)
            x = self.linear(x)
            if self.transposed: x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.stride)
            state = list(torch.unbind(x, dim=-1))
        else: assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.stride), device=device) # (batch, h, s)
        state = list(torch.unbind(state, dim=-1)) # List of (..., H)
        return state

    @property
    def d_output(self): return self._d_output

registry = {
    'sample': DownSample,
    'pool': DownAvgPool,
    'avg': DownAvgPool,
    'linear': DownLinearPool,
    'spectral': DownSpectralPool,
}

up_registry = {
    # 'sample': UpSample,
    'pool': UpAvgPool,
    'avg': UpAvgPool,
    'linear': UpLinearPool,
    # 'spectral': UpSpectralPool, # Not implemented and no way to make this causal
}
