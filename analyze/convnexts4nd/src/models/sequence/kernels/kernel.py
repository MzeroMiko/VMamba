"""Construct wide convolution kernels."""

from typing import Optional, Mapping, Tuple, Union
from collections import defaultdict
import math
import torch
import torch.nn as nn

import src.utils.train
log = src.utils.train.get_logger(__name__)


class Kernel(nn.Module):
    """Interface for modules that produce convolution kernels.

    A main distinction between these and normal Modules is that the forward pass
    does not take inputs. It is a mapping from parameters to a tensor that can
    be used in other modules, in particular as a convolution kernel.

    Because of the unusual parameterization, these kernels may often want special
    hyperparameter settings on their parameters. The `register` method provides
    an easy interface for controlling this, and is intended to be used with an
    optimizer hook that can be found in train.py or example.py.

    This class also defines an interface for interacting with kernels *statefully*,
    in particular for state space models (SSMs). This interface handles the setting
    when a model can be converted from a "CNN" into an "RNN".
    _setup_step()
    step()
    default_state()
    forward_state()

    See ConvKernel for the simplest instantiation of this interface.
    """

    def __init__(
        self,
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        **kwargs,
    ):
        """General interface.

        d_model (H): Model dimension, or number of independent convolution kernels created.
        channels (C): Extra dimension in the returned output (see .forward()).
            - One interpretation is that it expands the input dimension giving it C separate "heads" per feature.
              That is convolving by this kernel maps shape (B L D) -> (B L C D)
            - This is also used to implement a particular form of bidirectionality in an efficient way.
            - In general for making a more powerful model, instead of increasing C
              it is recommended to set channels=1 and adjust H to control parameters instead.
        l_max (L): Maximum kernel length (optional). If unspecified, most Kernel instantiations
            will return kernels of arbitrary length as passed into .forward().
        lr: Optional dictionary specifying special hyperparameters for .register().
            Passing in a number (e.g. 0.001) sets attributes of SSM parameters (A, B, dt).
            A custom optimizer hook is needed to configure the optimizer to set the learning rates appropriately for these parameters.
        wd: Same as lr, but for weight decay.
        """
        super().__init__()
        assert d_model > 0
        self.H = self.d_model = d_model
        self.L = self.l_max = l_max
        self.channels = channels
        self.lr = lr
        self.wd = wd
        self.verbose = verbose

        # Add a catch-all **kwargs to make it easier to change kernels
        # without manually moving other options passed in the config.
        # Good to log these just so it's explicit.
        if self.verbose and len(kwargs) > 0:
            log.info(f"{type(self)} extra kwargs: {kwargs}")

        # Logic for registering parameters
        # Case 1: lr: None | float
        #   All params should have this lr (None means inherit from global lr)
        # Case 2: lr: dict
        #   Specified params should have that lr, all others should be None
        if self.lr is None or isinstance(self.lr, float):
            self.lr_dict = defaultdict(lambda: self.lr)
        else:
            self.lr_dict = defaultdict(lambda: None)
            self.lr_dict.update(self.lr)

        # Same logic for weight decay
        # (but is always just set to 0.0 and hasn't been ablated)
        if self.wd is None or isinstance(self.wd, float):
            self.wd_dict = defaultdict(lambda: self.wd)
        else:
            self.wd_dict = defaultdict(lambda: None)
            self.wd_dict.update(self.wd)

    def forward(self, state=None, rate=1.0, L=None):
        """General interface to generate a global convolution kernel.

        state: Initial state for recurrent updates.
            E.g. for SSMs, this should have shape (B, H, N) (batch, d_model, d_state).
        rate: Relative sampling rate.
        L: Target kernel length.

        Returns:
          - (C, H, L) (channels, d_model, l_kernel) The convolution kernel.
          - (B, H, L) (batch, d_model, l_kernel)
              Extra information for how the state affects the output of convolving by kernel.
        """
        raise NotImplementedError

    def register(self, name, tensor, lr=None, wd=0.0):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)

    def _setup_step(self, **kwargs):
        """Convert a model into a recurrent mode for autoregressive inference."""
        raise NotImplementedError

    def step(self, x, state, **kwargs):
        """Step the model for one timestep with input x and recurrent state."""
        raise NotImplementedError

    def default_state(self, *args, **kwargs):
        """Return a default initial state."""
        raise NotImplementedError

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through the kernel."""
        raise NotImplementedError

    @property
    def d_state(self):
        """Implement this for interfaces that want to interact with a stateful layer (i.e. SSMs).

        Currently the only codepath that might use this is the StateDecoder, which is not used.
        """
        raise NotImplementedError

    @property
    def state_to_tensor(self):
        """Same as d_state, only needed for niche codepaths involving recurrent state."""
        raise NotImplementedError

class ConvKernel(Kernel):
    """Baseline implemented as a free convolution kernel."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.L is not None

        kernel = torch.randn(self.channels, self.H, self.L) / (self.H*self.L)**0.5
        # Register parameters
        self.register("kernel", kernel, self.lr_dict['K'], self.wd_dict['K'])

    def forward(self, state=None, rate=1.0, L=None):
        return self.kernel, None

class EMAKernel(Kernel):
    """Translation of Mega's MultiHeadEMA.

    This is a minimal implementation of the convolution kernel part of the module.
    This module, together with the main S4 block in src.models.sequence.modules.s4block
    (which is really just a fft-conv wrapper around any convolution kernel,
    such as this one), should be exactly equivalent to using the original Mega
    EMA module in src.models.sequence.modules.megablock.

    Two additional flags have been provided to resolve discrepencies in parameter
    count between S4(D) and EMA
    - `dt_tie` makes the shape of the step size \\Delta (H, 1) instead of (H, N)
    - `efficient_bidirectional` ties the A/B/dt parameters for the conv kernels
      in both forwards and backwards directions. This should have exactly the same
      speed, slightly more parameter efficiency, and similar performance.
    """

    def __init__(
        self,
        d_state: int = 2,
        dt_tie: bool = False,
        efficient_bidirectional: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.N = N = d_state
        self.channels = self.channels
        self.scale = math.sqrt(1.0 / self.N)

        # Exactly match the parameter count of S4(D) when bididirectional is on
        self.efficient_bidirectional = efficient_bidirectional
        if self.efficient_bidirectional:
            H_C = self.H * self.channels
        else:
            self.H *= self.channels
            H_C = self.H

        delta = torch.Tensor(self.H, 1 if dt_tie else N, 1)
        alpha = torch.Tensor(self.H, N, 1)
        beta = torch.Tensor(self.H, N, 1)
        self.register("delta", delta, self.lr_dict['dt'], self.wd_dict['dt'])
        self.register("alpha", alpha, self.lr_dict['dt'], self.wd_dict['dt'])
        self.register("beta", beta, self.lr_dict['dt'], self.wd_dict['dt'])
        self.gamma = nn.Parameter(torch.Tensor(H_C, N))
        # D skip connection handled by outside class
        # self.omega = nn.Parameter(torch.Tensor(H))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # Mega comment: beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.N, 1)
            if self.N > 1:
                idx = torch.tensor(list(range(1, self.N, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            # nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def coeffs(self): # Same as discretize
        p = torch.sigmoid(self.delta)  # (H N 1)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def forward(self, L=None, state=None, rate=1.0):
        L = L if self.l_max is None else min(self.l_max, L)
        p, q = self.coeffs()  # (H N 1)
        vander = torch.arange(L).to(p).view(1, 1, L) * torch.log(q)  # (H N L)
        kernel = (p * self.beta) * torch.exp(vander)
        if self.efficient_bidirectional:
            C = rearrange(self.gamma * self.scale, '(c h) n -> c h n', c=self.channels)
            kernel = torch.einsum('dnl,cdn->cdl', kernel, C)
        else:
            kernel = torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)
            kernel = rearrange(kernel, '(c h) l -> c h l', c=self.channels)

        kernel = kernel[..., :L]
        return kernel, None
