"""Implements variant of HiPPO-RNN that doesn't feed the hidden and memory states into each other time-wise, instead using simpler linear recurrences in time and letting them interact depthwise.

[21-10-22] AG: This was old experimental code. It should still work (perhaps with some minimal modifications), but there is not much reason to use this now. This was the initial step toward "deep linear parallelizable" versions of the HiPPO RNN which culminated in LSSL and S3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal

from src.models.nn import LinearActivation
from src.models.functional import unroll
from src.models.hippo.hippo import transition
from src.models.hippo.transition import TLagTAdaptiveTransitionManual, LagTAdaptiveTransitionManual, LegTAdaptiveTransitionManual, LegSAdaptiveTransitionManual, LagTCumsumAdaptiveTransition, TLagTCumsumAdaptiveTransition
from src.models.sequence.base import SequenceModule

class MemoryProjection(nn.Module):
    """Implements the memory projection operator for fixed dt."""

    def __init__(self, order, measure, dt, discretization='bilinear'):
        super().__init__()
        self.order = order
        A, B = transition(measure, order)
        C = np.ones((1, order))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)


        self.register_buffer('A', torch.Tensor(A))
        self.register_buffer('B', torch.Tensor(B))

    def forward(self, inputs):
        """
        inputs : (length, batch, size)
        output : (length, batch, size, order)
        # TODO this puts the unsqueeze inside here rather than outside, should make RNN versions the same
        """

        # L, B, S = inputs.shape
        inputs = inputs.unsqueeze(-1)
        u = F.linear(inputs, self.B)
        # output = unroll.unroll(self.A, u)
        output = unroll.parallel_unroll_recursive(self.A, u)
        # output = unroll.parallel_unroll_iterative(self.A, u)
        # output = unroll.variable_unroll(self.A, u, variable=False)

        # m = inputs.new_zeros(B, S, self.order)
        # outputs = []
        # for input in torch.unbind(inputs, dim=0):
        #     m = m + F.linear(m, self.A) + F.linear(input, self.B)

        # output = torch.stack(outputs, dim=0)
        return output

class VariableMemoryProjection(nn.Module):
    """Version of MemoryProjection with variable discretization.

    Materializes the transition matrices.
    """

    def __init__(self, order=1, measure='legs', dt=None):
        super().__init__()
        self.order = order
        self.measure = measure
        self.dt = dt

        # TODO incorporate measure
        if self.measure == 'legs':
            self.transition = LegSAdaptiveTransitionManual(self.order)
        elif self.measure == 'legt':
            self.transition = LegTAdaptiveTransitionManual(self.order)
        elif self.measure == 'lagt':
            self.transition = LagTAdaptiveTransitionManual(self.order)
        elif self.measure == 'tlagt':
            self.transition = TLagTAdaptiveTransitionManual(self.order)
        else:
            assert False, f"VariableMemoryProjection: measure {measure} not allowed"

        # Cached tensors
        self.register_buffer('I', torch.eye(self.order))
        self.register_buffer('zero', torch.zeros(self.order, self.order))

    def forward(self, inputs, dt=None):
        """
        inputs : (L, B, M)
        dt     : (L, B, M)
        output : (L, B, M, N) [length, batch, size, order]
        # TODO this puts the input unsqueeze inside here rather than outside, should make RNN versions the same
        """

        L, B, M = inputs.shape

        # Construct discretization if necessary
        if dt is None:
            if self.dt is None:
                dt = torch.cumsum(inputs.new_ones(L), dim=0) # no new_arange
                dt = (1./dt)[:, None, None] # (L, 1, 1)
            else:
                dt = torch.full((L, 1, 1), self.dt).to(inputs) # fixed dt

        # Create transition matrices
        # I = self.I[:, None, None, None, :].expand((self.order, L, B, M, self.order)) # (N, L, B, M, N)
        I = self.I[:, None, None, None, :].repeat((1, L, B, M, 1)) # (N, L, B, M, N)
        As = self.transition.bilinear(dt, I, 0) # (N, L, B, M, N) # NOTE due to the broadcasting here, the ManualTransition actually swaps axes back for efficiency; can potential save if this axis reordering is too slow [probably not a bottleneck]
        As = As.permute((1, 2, 3, 0, 4)) # (L, B, M, N, N)
        # TODO this A might be transposed; should print to compare
        # print(As.shape)
        Bs = self.transition.bilinear(dt, inputs.new_zeros(self.order), 1) # (L, B, M, N)

        inputs = inputs.unsqueeze(-1)
        # u = F.linear(inputs, self.transition.B) # (L, B, M, N)
        # u = F.linear(inputs, Bs) # (L, B, M, N)
        u = inputs * Bs # (L, B, M, N)
        # output = unroll.unroll(self.A, u)
        # output = unroll.parallel_unroll_recursive(self.A, u)
        output = unroll.variable_unroll(As, u, variable=True)
        # output = unroll.parallel_unroll_iterative(self.A, u)

        return output

class ToeplitzMemoryProjection(nn.Module):
    def __init__(self, order, measure, measure_args={}):
        super().__init__()
        self.N = order

        if measure == 'lagt':
            self.transition = LagTCumsumAdaptiveTransition(self.N)
        elif measure == 'glagt':
            # TODO this is broken
            alpha = measure_args.get('alpha', 0.0)
            beta = measure_args.get('beta', 0.01)
            self.transition = GLagTCumsumAdaptiveTransition(self.N, alpha, beta)
        else:
            assert False, f"ToeplitzMemoryProjection: measure {measure} not supported"

        e = torch.zeros(self.N)
        e[0] = 1
        self.register_buffer('e', e) # the e_0 basis vector

    def forward(self, inputs, dt):
        """
        inputs : (L, B, M)
        dt     : (L, B, M)
        output : (L, B, M, N) [length, batch, size, order]
        # TODO this puts the unsqueeze inside here rather than outside, should make RNN versions the same
        """

        L, B, M = inputs.shape
        I = self.e.repeat((L, B, M, 1)) # (L, B, M, N)
        # I = self.e.repeat(inputs.shape+(1,)) # (L, B, M, N)
        As = self.transition.bilinear(dt, I, torch.zeros_like(dt)) # (L, B, M, N)
        # Bs = self.transition.bilinear(dt, torch.zeros_like(I), torch.ones_like(dt)) # (L, B, M, N)
        Bs = self.transition.bilinear(dt, torch.zeros_like(I), inputs) # (L, B, M, N)
        output = unroll.variable_unroll_toeplitz(As, Bs, pad=False)
        # print("HERE")
        return output


class HiPPOQRNN(SequenceModule):
    # TODO dropout?
    def __init__(
            self,
            d_input, d_model=256, memory_size=1, memory_order=-1,
            variable=False, dt=0.01,
            measure='lagt', measure_args={},
            dropout=0.0,
        ):
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError("Dropout currently not supported for QRNN")

        if memory_order < 0:
            memory_order = d_model
        self.d_input   = d_input
        self.d_model  = d_model
        self.memory_size  = memory_size
        self.memory_order = memory_order
        self.variable     = variable
        self.dt           = dt

        # TODO deal with initializers

        preact_ctor = LinearActivation
        preact_args = [self.d_input + self.memory_size * self.memory_order, self.d_model, True]
        self.W_hmx = preact_ctor(*preact_args)

        if self.variable:
            self.W_uh = nn.Linear(self.d_input, 2*self.memory_size)
            if measure in ['lagt', 'tlagt']:
                self.memory_proj = ToeplitzMemoryProjection(memory_order, measure, measure_args)
            else:
                self.memory_proj = VariableMemoryProjection(memory_order, measure)
        else:
            self.W_uh = nn.Linear(self.d_input, self.memory_size)
            self.memory_proj = MemoryProjection(memory_order, measure, dt)

        self.hidden_activation_fn = torch.tanh
        self.memory_activation_fn = nn.Identity()

    # @profile
    def forward(self, inputs, return_output=False):
        """
        inputs : [length, batch, dim]
        """
        L, B, d_input = inputs.shape
        assert d_input == self.d_input

        u = self.memory_activation_fn(self.W_uh(inputs)) # (L, B, memory_size)

        if self.variable:
            # Automatic scaling dt
            M = self.memory_size
            # dt = torch.full((L, 1, 1), self.dt).to(inputs) # fixed dt to test
            dt = torch.sigmoid(u[..., M:]) # variable dt
            u = u[..., :M]
            m = self.memory_proj(u, dt)
        else:
            m = self.memory_proj(u) # (L, B, M, N)


        mx = torch.cat((m.view(L, B, self.memory_size*self.memory_order), inputs), dim=-1) # length, batch, d_input
        h = self.hidden_activation_fn(self.W_hmx(mx)) # length, batch, d_model


        if return_output:
            return h, h[-1, ...]
        else:
            return None, h[-1, ...]

    def default_state(self, x, batch_shape):
        raise NotImplementedError("Needs to be implemented.")

    def step(self, x, state):
        raise NotImplementedError("Needs to be implemented.")

    @property
    def d_state(self):
        return self.d_model

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return lambda state: state
