"""Implementation of LSSL module. Succeeded by S4."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import DictConfig

from src.models.nn import Activation
from src.models.functional.krylov import krylov
from src.models.hippo import transition, hippo
from src.models.functional.toeplitz import causal_convolution
from src.models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U

def linear_system_from_krylov(u, C, D, k):
    """
    Computes the state-space system y = Cx + Du from Krylov matrix K(A, B)

    u: (L, B, ...) ... = H
    C: (..., M, N) ... = H
    D: (..., M)
    k: (..., N, L) Krylov matrix representing b, Ab, A^2b...

    y: (L, B, ..., M)
    """


    # Equivalent ways to perform C @ k, slight speed differences
    k = C @ k # (..., M, L)
    # k = torch.einsum('... m n, ... n l -> ... m l', C, k) # C @ k
    # k = torch.sum(k.unsqueeze(-3) * C.unsqueeze(-1), dim=-2) # (..., M, L) C @ k

    k = rearrange(k, '... m l -> m ... l')
    k = k.to(u) # if training in half precision, need to go back to float32 for the fft
    k = k.unsqueeze(1) # (M, 1, ..., L)

    v = u.unsqueeze(-1).transpose(0, -1) # (1, B, ..., L)
    y = causal_convolution(k, v, fast=True) # (M, B, ..., L)
    y = y.transpose(0, -1) # (L, B, ..., M)
    y = y + u.unsqueeze(-1) * D # (L, B, ..., M)
    return y

class Platypus(SequenceModule):
    """ Implementation of LSSL module.
    # TODO this expects (length, batch) but this codebase is now (batch, length)
    """
    requires_length = True

    def __init__(
            self,
            d,
            d_model=-1, # overloading this term, same as memory_order or N
            measure='legs', # 'legs', 'legt' main ones; can also try 'lagt'
            measure_args={},
            learn=0, # 0 means no learn, 1 means same A matrix for each hidden feature H, 2 means different A matrix per feature. 1 does not change parameter count. 2 adds parameters but does not slow down
            lr=0.0001, # controls learning rate of transition parameters
            noise=0.0, # injects input noise to the state space system
            init='normal', # for debugging, but might be useful?
            dt=None,
            channels=1, # denoted by M below
            bias=False,
            activation='gelu',
            ff=True,
            weight_norm=False,
            dropout=0.0,
            l_max=-1,
        ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.d = d
        self.N = d_model if d_model > 0 else d
        self.dt = DictConfig({
            'min' : 0.001,
            'max' : 0.1,
            'learn' : False,
            'lr': 0.001,
            'init' : 'random',
        })
        if dt is not None: self.dt.update(dt)
        self.ff = ff
        self.bias = bias


        # Construct transition
        self.learn = learn
        if self.learn == 0:
            if measure == 'identity': # for testing
                A, B = torch.eye(self.N), torch.ones(self.N)
                self.transition = transition.ManualAdaptiveTransition(self.N, A, B)
            elif measure == 'random':
                A = torch.randn(self.N, self.N) / self.N # E[AA^T] = (1/N)I -- empirically I nans out
                B = torch.ones(self.N) # based on HiPPO matrices; worth trying random, haven't tried
                self.transition = transition.ManualAdaptiveTransition(self.N, A, B)
            elif measure == 'legt':
                # self.transition = transition.LegTAdaptiveTransition(self.N)
                self.transition = transition.LegTTriDInverseAdaptiveTransition(self.N, **measure_args)
            elif measure == 'cheb':
                self.transition = transition.ChebITriDInverseAdaptiveTransition(self.N, **measure_args)
            elif measure == 'chebii':
                self.transition = transition.ChebIITriDInverseAdaptiveTransition(self.N, **measure_args)
            elif measure == 'lagt':
                self.transition = transition.LagTCumsumAdaptiveTransition(self.N, **measure_args)
            elif measure == 'glagt':
                self.transition = transition.GLagTToeplitzAdaptiveTransition(self.N, **measure_args)
            elif measure == 'legs':
                self.transition = transition.LegSTriDInverseAdaptiveTransition(self.N, **measure_args)
            elif measure == 'jac':
                self.transition = transition.JacTriDInverseAdaptiveTransition(self.N, **measure_args)
            else:
                raise NotImplementedError
        elif self.learn == 1 or self.learn == 2:
            kwargs = {'trainable': True, 'lr': lr}
            kwargs.update(measure_args)
            if self.learn == 2:
                kwargs['batch'] = (self.d,)
            if measure == 'random':
                A = torch.randn(self.N, self.N) / self.N # E[AA^T] = (1/N)I . empirically I doesn't work, dunno why
                B = torch.ones(self.N) # based on HiPPO matrices; worth trying random, haven't tried
                self.transition = transition.ManualAdaptiveTransition(self.N, A, B, **kwargs)
            elif measure == 'legt':
                self.transition = transition.LegTTriDInverseAdaptiveTransition(self.N, **kwargs)
            elif measure == 'lagt':
                self.transition = transition.LagTTriDInverseAdaptiveTransition(self.N, **kwargs)
            elif measure == 'legs':
                self.transition = transition.LegSTriDInverseAdaptiveTransition(self.N, **kwargs)
            elif measure == 'cheb':
                self.transition = transition.ChebITriDInverseAdaptiveTransition(self.N, **kwargs)
            elif measure == 'chebii':
                self.transition = transition.ChebIITriDInverseAdaptiveTransition(self.N, **kwargs)
            elif measure == 'toep':
                self.transition = transition.LagTToeplitzAdaptiveTransition(self.N, **kwargs)
            else: raise NotImplementedError
        elif self.learn == 3: # for debugging
            A, B = hippo.transition(measure, self.N)
            B = B[:, 0]
            self.transition = transition.ManualAdaptiveTransition(self.N, A, B, trainable=True, lr=lr)
        else:
            raise NotImplementedError


        self.m = channels

        if init == 'normal':
            self.C = nn.Parameter(torch.randn(self.d, self.m, self.N))
            self.D = nn.Parameter(torch.randn(self.d, self.m))
        elif init == 'constant':
            self.C = nn.Parameter(torch.ones(self.d, self.m, self.N))
            self.D = nn.Parameter(torch.ones(self.d, self.m))
        elif init == 'uniform':
            self.C = nn.Parameter(1.732 * torch.rand(self.d, self.m, self.N))
            self.D = nn.Parameter(torch.randn(self.d, self.m))
        else: raise NotImplementedError
        if self.bias:
            self.E = nn.Parameter(torch.zeros(self.d, self.m))

        if self.dt.init == 'uniform':
            log_dt = torch.linspace(math.log(self.dt.min), math.log(self.dt.max), self.d)
        elif self.dt.init == 'random':
            log_dt = torch.rand(self.d) * (math.log(self.dt.max)-math.log(self.dt.min)) + math.log(self.dt.min)
        else: raise NotImplementedError

        if self.dt.learn:
            self.log_dt = nn.Parameter(log_dt) # (H)
            self.log_dt._lr = self.dt.lr # register the parameter for the optimizer to reduce lr
        else:
            self.register_buffer('log_dt', log_dt)
        self.k = None
        self.noise = noise

        self.activate = Activation(activation)
        self.drop = nn.Dropout(dropout)

        if self.ff:
            self.output_linear = nn.Linear(self.m * self.d, self.d)

            if weight_norm:
                self.output_linear = nn.utils.weight_norm(self.output_linear)

        # For test time shift
        self.l_max = l_max
        self.last_len = -1

    def forward(self, u, *args, state=None, **kwargs):
        """
        u: (L, B, H) [21-09-29] Our backbone now passes inputs as (B, L, H). This calss originally expected (L, B, H) so we transpose accordingly
        state: (B, H, N) previous hidden state of the recurrence
        """
        next_state = None

        u = u.transpose(0, 1)

        # Construct dt (H)
        dt = torch.exp(self.log_dt) # Note: if dt is not learnable this slightly wastes computation, but it isn't a bottleneck

        ## # Calculate test-time shift
        # changed sampling rate; uncache Krylov
        if self.last_len != u.shape[0]:
            self.k = None
            self.last_len = u.shape[0]
        # Calculate change from train sampling rate
        if self.l_max > 0:
            rate = self.l_max / u.shape[0]
            # if rate != 1.0: dt = dt * rate
            if rate != 1.0: rate = round(rate)
            else: rate = None
        else:
            rate = None


        # We need to compute the "recurrence" if
        # (*) there is noise or an initial state
        # (*) we're learning the system A, B
        # (*) first pass
        kb = [] # will store the B vectors for Krylov computation
        _learn = (self.dt.learn or self.learn) and self.training # need to learn and it's training time # TODO this ignores the last training minibatch if no test time shift (prev batch's K gets cached)... should recalculate A in the last_len check ideally
        _conv = _learn or self.k is None or u.shape[0] > self.k.shape[-1] # or rate
        _noise = self.noise > 0.0 and self.training
        if _conv:
            B = self.transition.gbt_B(dt) # (..., N) depending if learn=2
            kb.append(B)
        if _noise:
            noise = self.noise * torch.randn(self.d, self.N, dtype=u.dtype, device=u.device) # (H, N)
            kb.append(noise)

        A = None
        if len(kb) > 0:
            if rate is not None:
                dt = dt * rate

            A = self.transition.gbt_A(dt) # (..., N, N) (..., N)

            # Adjust by rate
            # if _conv and rate is not None:
            #     while rate > 1:
            #         B = B + torch.sum(A * B.unsqueeze(-2), dim=-1) # (I + A) @ B
            #         A = A @ A
            #         rate //= 2

            kb = [b.broadcast_to(dt.shape+(self.N,)) for b in kb]
            kb = torch.stack(torch.broadcast_tensors(*kb), dim=0) # each (..., N)
            krylovs = krylov(u.shape[0], A, kb) # (H, N, L) each
            k_noise, k_conv = torch.split(
                krylovs,
                split_size_or_sections=[int(_noise), int(_conv)],
                dim=0
            )
            if _conv: # Cache the Krylov matrix K(A, B)
                self.k = k_conv.squeeze(0) # (H, N, L)
            if _noise:
                k_noise = k_noise.squeeze(0) # (H, N, L)

        # Convolution
        y = linear_system_from_krylov(u, self.C, self.D, self.k[..., :u.shape[0]]) # (L, B, H, M)
        if _noise:
            k_noise = torch.cumsum(k_noise, dim=-1) # (H, N, L) w + Aw + A^2w + ...
            k_noise = contract('h m n, h n l -> l h m', self.C, k_noise) # C @ k
            y = y + k_noise.unsqueeze(1) # (L, B, H, M)
            y = y + self.noise * torch.randn(y.shape, dtype=u.dtype, device=u.device)

        # State needs a special case because it has a batch dimension
        if state is not None: # (B, H, N)
            if A is None: A = self.transition.gbt_A(dt) # (..., N, N) (..., N)

            ATC, ATL = krylov(u.shape[0], A.transpose(-1,-2), self.C.transpose(0, 1), return_power=True) # (M, H, N, L), (H, N, N) represents A^T C and (A^T)^L
            y = y + contract('mhnl, bhn -> lbhm', ATC, state)

            # Compute next state
            with torch.no_grad():
                next_state = contract('hnp, bhn -> bhp', ATL, state)
                if _noise:
                    next_state = next_state + k_noise[..., -1]
                next_state = next_state + contract('lbh, hnl -> bhn', u.flip(0), self.k[:..., u.shape[0]]) # (B, H, N)
                next_state = contract('hnp, bhp -> bhn', A, next_state)
                next_state = next_state.detach() # TODO necessary?

            # Debugging code useful for checking if state computation is correct
            # from models.functional.unroll import variable_unroll_sequential, variable_unroll
            # B = self.transition.gbt_B(dt)
            # inps = B*u.unsqueeze(-1) # (L, B, H, N)
            # inps[0] = inps[0] + state
            # xx = variable_unroll(A, inps, variable=False)
            # yy = torch.sum(self.C * xx.unsqueeze(-2), dim=-1)
            # yy = yy + u.unsqueeze(-1) * self.D # true output y; should equal y
            # xx_ = variable_unroll(A, B*u.unsqueeze(-1), variable=False)
            # yy_ = torch.sum(self.C * xx_.unsqueeze(-2), dim=-1)
            # yy_ = yy_ + u.unsqueeze(-1) * self.D # output without state; should equal y before the C A^T S term was added
            # ss = (A @ xx[-1].unsqueeze(-1)).squeeze(-1) # should equal next_state
            # breakpoint()
            # y = z

        # bias term
        if self.bias:
            y = y + self.E

        y = self.drop(self.activate(y))

        y = rearrange(y, 'l b h m -> l b (h m)') # (L, B, H*M)

        if self.ff:
            y = self.output_linear(y) # (L, B, H)
        y = y.transpose(0, 1) # Back to (B, L, H) as expected
        return y, next_state

    def is_initialized(self):
        return self.k is not None

    def initialize(self, shared_params):
        if 'k' in shared_params:
            self.k = shared_params['k']
        else:
            dt = torch.exp(self.log_dt)
            A = self.transition.gbt_A(dt) # (..., N, N)
            B = self.transition.gbt_B(dt) # (..., N)
            self.k = krylov(1024, A, B) # (L, H, N) each
            shared_params['k'] = self.k

    def default_state(self, *batch_shape, device=None):
        return torch.zeros(*batch_shape, self.N, device=device)

    def step(self, x, state):
        raise NotImplementedError("Needs to be implemented.")

    @property
    def d_state(self):
        return self.d

    @property
    def d_output(self):
        return self.d

    @property
    def state_to_tensor(self):
        return lambda state: state

LSSL = TransposedModule(Platypus)
