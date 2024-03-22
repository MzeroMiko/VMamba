"""The core RNN cell architecture of the HiPPO-RNN from the original HiPPO paper."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from scipy import signal
from scipy import linalg as la

from src.models.sequence.rnns.cells.basic import RNNCell
from src.models.nn import LinearActivation, Activation # , get_initializer
from src.models.nn.gate import Gate


forward_aliases   = ['euler', 'forward_euler', 'forward', 'forward_diff']
backward_aliases  = ['backward', 'backward_diff', 'backward_euler']
bilinear_aliases = ['bilinear', 'tustin', 'trapezoidal', 'trapezoid']
zoh_aliases       = ['zoh']


class MemoryCell(RNNCell):
    """This class handles the general architectural wiring of the HiPPO-RNN, in particular the interaction between the hidden state and the linear memory state.

    Specific variants can be instantiated by subclassing this with an appropriately defined update_memory() method.
    """

    name = None
    valid_keys = ['uxh', 'ux', 'uh', 'um', 'hxm', 'hx', 'hm', 'hh', 'bias', ]

    @property
    def default_initializers(self):
        return {
            'uxh': 'uniform',
            'hxm': 'xavier',
            'um': 'zero',
            'hh': 'xavier',
        }


    @property
    def default_architecture(self):
        return {
            'ux': True,
            'hx': True,
            'hm': True,
            'hh': False,
            'bias': True,
        }


    def __init__(
            self, d_input, d_model, memory_size, memory_order,
            memory_activation='id',
            gate='G', # 'N' | 'G' | UR'
            **kwargs
        ):
        self.memory_size       = memory_size
        self.memory_order      = memory_order

        self.memory_activation = memory_activation
        self.gate              = gate

        super(MemoryCell, self).__init__(d_input, d_model, **kwargs)


        self.input_to_d_model = self.d_input if self.architecture['hx'] else 0
        self.input_to_memory_size = self.d_input if self.architecture['ux'] else 0

        # Hidden to memory
        self.W_uxh = LinearActivation(
            self.input_to_memory_size + self.d_model,
            self.memory_size,
            bias=self.architecture['bias'],
            initializer=self.initializers['uxh'],
            activation=self.memory_activation,
            activate=True,
        )


        self.memory_to_d_model = self.memory_size * self.memory_order if self.architecture['hm'] else 0

        # Memory to hidden
        self.W_hxm = LinearActivation(
            self.input_to_d_model + self.memory_to_d_model,
            self.d_model,
            self.architecture['bias'],
            initializer=self.initializers['hxm'],
            activation=self.hidden_activation,
            activate=False,
        )

        if self.architecture['hh']:
            self.reset_hidden_to_hidden()
        else:
            self.W_hh = None

        # Construct gate with options
        if self.gate is not None:
            preact_ctor = LinearActivation
            preact_args = [
                self.input_to_d_model + self.memory_to_d_model,
                self.d_model,
                self.architecture['bias'],
            ]
            if self.architecture['hh']:
                print("input to hidden size, memory to hidden size, hidden size:", self.input_to_d_model, self.memory_to_d_model, self.d_model)
                preact_args[0] += self.d_model
            self.W_gxm = Gate(self.d_model, preact_ctor, preact_args, mechanism=self.gate)

    def reset_parameters(self):
        # super().reset_parameters() # TODO find a way to refactor to call super()
        self.activate = Activation(self.hidden_activation, self.d_model)

    def forward(self, input, state):
        h, m, time_step = state

        # Update the memory
        u = self.forward_memory(input, h, m)
        m = self.update_memory(m, u, time_step) # (batch, memory_size, memory_order)

        # Update hidden
        h = self.forward_hidden(input, h, m)

        next_state = (h, m, time_step + 1)
        output = self.state_to_tensor(next_state)

        return output, next_state


    def forward_memory(self, input, h, m):
        """First part of forward pass to construct the memory state update."""

        input_to_memory = input if self.architecture['ux'] else input.new_empty((0,))
        xh = torch.cat((input_to_memory, h), dim=-1)

        # Construct the update features
        u = self.W_uxh(xh)  # (batch, memory_size)
        return u

    def forward_hidden(self, input, h, m):
        input_to_hidden = input if self.architecture['hx'] else input.new_empty((0,))

        # Update hidden state from memory
        memory_to_hidden = m.view(input.shape[0], self.memory_size*self.memory_order)
        xm = torch.cat((input_to_hidden, memory_to_hidden), dim=-1)
        hidden_preact = self.W_hxm(xm)

        if self.architecture['hh']:
            hidden_preact = hidden_preact + self.W_hh(h)
        hidden = self.activate(hidden_preact)


        # Construct gate if necessary
        if self.gate is None:
            h = hidden
        else:
            if self.architecture['hh']:
                xm = torch.cat((xm, h), dim=-1)
            g = self.W_gxm(xm)
            h = (1.-g) * h + g * hidden

        return h


    def update_memory(self, m, u, time_step):
        """
        m: (B, M, N) [batch size, memory size, memory order]
        u: (B, M)

        Output: (B, M, N)
        """
        raise NotImplementedError

    def default_state(self, *batch_shape, device=None):
        return (
            torch.zeros(*batch_shape, self.d_model, device=device, requires_grad=False),
            torch.zeros(*batch_shape, self.memory_size, self.memory_order, device=device, requires_grad=False),
            0,
        )

    @property
    def state_to_tensor(self):
        """Converts a state into a single output (tensor)."""
        def fn(state):
            h, m, time_step = state
            return h
        return fn

    @property
    def d_state(self):
        return self.d_model

    @property
    def d_output(self):
        return self.d_model


class LTICell(MemoryCell):
    """A cell where the memory state follows Linear Time Invariant dynamics: c' = Ac + Bf."""

    def __init__(
            self, d_input, d_model, memory_size, memory_order,
            A, B,
            dt=0.01,
            discretization='zoh',
            **kwargs
        ):
        super().__init__(d_input, d_model, memory_size, memory_order, **kwargs)


        C = np.ones((1, memory_order))
        D = np.zeros((1,))
        dA, dB, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)


        dA = dA - np.eye(memory_order)  # puts into form: x += Ax
        self.register_buffer('A', torch.Tensor(dA))
        self.register_buffer('B', torch.Tensor(dB))

    def update_memory(self, m, u, time_step):
        u = u.unsqueeze(-1) # (B, M, 1)
        return m + F.linear(m, self.A) + F.linear(u, self.B)

class LSICell(MemoryCell):
    """A cell where the memory state Linear 'Scale' Invariant dynamics: c' = 1/t (Ac + Bf)."""

    def __init__(
            self, d_input, d_model, memory_size, memory_order,
            A, B,
            init_t = 0,  # 0 for special case at t=0 (new code), else old code without special case
            l_max=1024,
            discretization='bilinear',
            **kwargs
        ):
        # TODO: make init_t start at arbitrary time (instead of 0 or 1)

        # B should have shape (N, 1)
        assert len(B.shape) == 2 and B.shape[1] == 1

        super().__init__(d_input, d_model, memory_size, memory_order, **kwargs)

        assert isinstance(init_t, int)
        self.init_t = init_t
        self.l_max = l_max

        A_stacked = np.empty((l_max, memory_order, memory_order), dtype=A.dtype)
        B_stacked = np.empty((l_max, memory_order), dtype=B.dtype)
        B = B[:,0]
        N = memory_order
        for t in range(1, l_max + 1):
            At = A / t
            Bt = B / t
            if discretization in forward_aliases:
                A_stacked[t - 1] = np.eye(N) + At
                B_stacked[t - 1] = Bt
            elif discretization in backward_aliases:
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, np.eye(N), lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At, Bt, lower=True)
            elif discretization in bilinear_aliases:
                A_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, np.eye(N) + At / 2, lower=True)
                B_stacked[t - 1] = la.solve_triangular(np.eye(N) - At / 2, Bt, lower=True)
            elif discretization in zoh_aliases:
                A_stacked[t - 1] = la.expm(A * (math.log(t + 1) - math.log(t)))
                B_stacked[t - 1] = la.solve_triangular(A, A_stacked[t - 1] @ B - B, lower=True)
        B_stacked = B_stacked[:, :, None]

        A_stacked -= np.eye(memory_order)  # puts into form: x += Ax
        self.register_buffer('A', torch.Tensor(A_stacked))
        self.register_buffer('B', torch.Tensor(B_stacked))


    def update_memory(self, m, u, time_step):
        u = u.unsqueeze(-1) # (B, M, 1)
        t = time_step - 1 + self.init_t
        if t < 0:
            return F.pad(u, (0, self.memory_order - 1))
        else:
            if t >= self.l_max: t = self.l_max - 1
            return m + F.linear(m, self.A[t]) + F.linear(u, self.B[t])
