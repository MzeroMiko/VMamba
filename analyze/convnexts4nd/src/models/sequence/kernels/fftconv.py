"""Module for FFT convolution that accepts a flexible kernel parameterization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.sequence import SequenceModule
from src.models.sequence.kernels import registry as kernel_registry
from src.models.nn import Activation, DropoutNd

contract = torch.einsum

class FFTConv(SequenceModule):
    """Implements an FFT Convolution around a convolution kernel.

    d_model (H): Model dimension (in CNN terminology, this would be "channels").
    l_max (L): The maximum kernel length. Set l_max=None to always use a global kernel.
    channels: Can be interpreted as a number of "heads"; the SSM is a map from a 1-dim to C-dim sequence. It's not recommended to change this; instead, increase d_model for larger models.
    bidirectional: If True, convolution kernel will be two-sided.
    activation: Activation after the full convolution.
    transposed, dropout, tie_dropout: More general model options, see SequenceModule.
    mode: Which kernel algorithm to use. 'nplr' is the full S4 model; 'diag' is the simpler S4D. Other options can be found in the kernel registry.

    kernel_args: See the class .kernel.SSMKernel for the kernel constructor which accepts kernel_args. Relevant options that are worth considering and tuning include "mode", "init", "dt_min", "dt_max", "lr"
    """

    def __init__(
        self,
        d_model,
        l_max=None,
        channels=1,
        swap_channels=False,
        bidirectional=False,
        activation=None, # Activation after layer
        transposed=True,
        dropout=0.0,
        tie_dropout=False,
        drop_kernel=0.0,
        mode='dplr',
        kernel=None,
        **kernel_args,  # Arguments passed into inner convolution kernel
    ):
        super().__init__()
        self.d_model = d_model
        self.L = self.l_max = l_max
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed
        self.swap_channels = swap_channels


        if activation is not None and activation.startswith('glu'):
            channels *= 2
        self.activation = Activation(activation, dim=1 if self.transposed else -1)

        self.D = nn.Parameter(torch.randn(channels, self.d_model))

        if self.bidirectional:
            channels *= 2

        # Inner convolution kernel
        if mode is not None:
            assert kernel is None, "Pass either mode or kernel but not both"
            # log.info(
            #     "Argument 'mode' is deprecated and renamed to 'kernel',"
            #     "and will be removed in a future version."
            # )
            kernel, mode = mode, kernel
        kernel_cls = kernel_registry[kernel]
        self.kernel = kernel_cls(
            d_model=self.d_model,
            l_max=self.l_max,
            channels=channels,
            **kernel_args,
        )

        dropout_fn = DropoutNd if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_kernel = nn.Dropout(drop_kernel) if drop_kernel > 0.0 else nn.Identity()

    def forward(self, x, state=None, rate=1.0, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B D L) if self.transposed else (B L D)
        """

        # Always work with (B D L) dimension in this module
        if not self.transposed: x = x.transpose(-1, -2)
        L = x.size(-1)

        # Compute SS Kernel
        l_kernel = L if self.L is None else min(L, round(self.L / rate))
        k, k_state =  self.kernel(L=l_kernel, rate=rate, state=state) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0))
            # The above has an off-by-one in the reverse direction
            # This is a deliberate choice since the off-by-one should not affect any applications
            # This can be amended which may be very slightly slower
            # k = F.pad(k0, (0, L)) \
            #         + F.pad(k1[..., 1:].flip(-1), (L+1, 0)) \
            #         + F.pad(k1[..., :1], (0, l_kernel+L-1))

        # Kernel dropout
        k = self.drop_kernel(k)

        # In principle, we could pad to l_kernel+L-1 instead of l_kernel+L, but we choose the latter for
        # equational simplicity. Additionally, we have not experimented to compare the efficiency of the two.
        k_f = torch.fft.rfft(k, n=l_kernel+L) # (C H L)
        x_f = torch.fft.rfft(x, n=l_kernel+L) # (B H L)
        y_f = contract('bhl,chl->bchl', x_f, k_f)
        y = torch.fft.irfft(y_f, n=l_kernel+L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', x, self.D)

        # Compute state update
        if state is not None:
            assert not self.bidirectional, "Bidirectional not supported with state forwarding"
            y = y + k_state #
            next_state = self.kernel.forward_state(x, state)
        else:
            next_state = None


        # Reshape to flatten channels
        if self.swap_channels:
            y = rearrange(y, 'b c h l -> b (h c) l')
        else:
            y = rearrange(y, 'b c h l -> b (c h) l')

        y = self.drop(y)  # DropoutNd better with transposed=True

        if not self.transposed: y = y.transpose(-1, -2)
        y = self.activation(y)

        return y, next_state


    def setup_step(self, **kwargs):
        self.kernel._setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        y, next_state = self.kernel.step(x, state) # (B C H)
        y = y + x.unsqueeze(-2) * self.D
        y = rearrange(y, 'b c h -> b (c h)')
        y = self.activation(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.kernel.d_state

    @property
    def d_output(self):
        return self.d_model * self.channels

    @property
    def state_to_tensor(self):
        return self.kernel.state_to_tensor
