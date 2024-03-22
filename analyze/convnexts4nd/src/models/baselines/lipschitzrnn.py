"""Adapted from LipschitzRNN https://github.com/erichson/LipschitzRNN.

Original code left as comments
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.models.sequence.base import SequenceModule

from copy import deepcopy

from torchdiffeq import odeint as odeint

def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]),
                                         torch.Tensor([std / n_units]))
    A_init = sampler.sample((n_units, n_units))[..., 0]
    return A_init


class LipschitzRNN_ODE(nn.Module):
    """The derivative of the continuous-time RNN, to plug into an integrator."""

    def __init__(self, d_model, beta, gamma, init_std):
        super().__init__()
        self.device = get_device()

        self.gamma = gamma
        self.beta = beta

        self.tanh = nn.Tanh()

        self.z = torch.zeros(d_model)
        self.C = nn.Parameter(gaussian_init_(d_model, std=init_std))
        self.B = nn.Parameter(gaussian_init_(d_model, std=init_std))
        self.I = torch.eye(d_model).to(self.device)
        self.i = 0

    def forward(self, t, h):
        """dh/dt as a function of time and h(t)."""
        if self.i == 0:
            self.A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                1 - self.beta) * (self.B +
                                  self.B.transpose(1, 0)) - self.gamma * self.I
            self.W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                1 - self.beta) * (self.C +
                                  self.C.transpose(1, 0)) - self.gamma * self.I

        return torch.matmul(
            h, self.A) + self.tanh(torch.matmul(h, self.W) + self.z)


class RnnModels(SequenceModule): #(nn.Module):
    """Generator of multiple possible general RNN forms."""
    @property
    def d_output(self): #TODO: check
        return self.d_model

    def __init__(self,
                #  d_input,
                #  d_output,
                 d_model=128,
                 chunk=1,
                 eps=0.01,
                 beta=0.8,
                 gamma=0.01,
                 gated=False,
                 init_std=1,
                 alpha=1,
                 model='LipschitzRNN',
                 solver='euler',
                 l_output=0,
                 l_max=-1,
                 ):
        super().__init__()

        # self.d_input = d_input
        self.d_model = d_model
        # self.chunk = chunk
        self.eps = eps
        self.model = model
        self.solver = solver
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha


        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # self.E = nn.Linear(d_input*self.chunk, d_model)
        # self.D = nn.Linear(d_model, d_output)
        self.register_buffer('I', torch.eye(d_model))

        if self.model == 'simpleRNN':
            self.W = nn.Linear(d_model, d_model, bias=False)
            self.W.weight.data = gaussian_init_(d_model, std=init_std)

        elif self.model == 'resRNN':
            self.W = nn.Linear(d_model, d_model, bias=False)
            self.W.weight.data = gaussian_init_(d_model, std=init_std)

        elif self.model == 'asymRNN':
            self.C = nn.Parameter(gaussian_init_(d_model, std=init_std))

        elif self.model == 'calRNN':
            U, _, V = torch.svd(gaussian_init_(d_model, std=init_std))
            self.C = nn.Parameter(torch.mm(U, V.t()).float())

        elif self.model == 'LipschitzRNN':
            self.C = nn.Parameter(gaussian_init_(d_model, std=init_std))
            self.B = nn.Parameter(gaussian_init_(d_model, std=init_std))

        elif self.model == 'LipschitzRNN_gated':
            self.C = nn.Parameter(gaussian_init_(d_model, std=init_std))
            self.B = nn.Parameter(gaussian_init_(d_model, std=init_std))
            # self.E_gate = nn.Linear(d_input, d_model)

        elif self.model == 'LipschitzRNN_ODE':
            self.func = LipschitzRNN_ODE(d_model, beta, gamma, init_std)

        else:
            print("Unexpected model!")
            raise NotImplementedError

    def step(self, x, state):
        # THIS CODE IS UNTESTED
        if self.model == 'LipschitzRNN':
            if state is None:
                A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                        1 - self.beta) * (self.B + self.B.transpose(
                            1, 0)) - self.gamma * self.I
                W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                    1 - self.beta) * (self.C + self.C.transpose(
                        1, 0)) - self.gamma * self.I
            state = state + self.eps * self.alpha * torch.matmul(state, A) + \
                self.eps * self.tanh(torch.matmul(state, W) + x)
        return x, state

    def forward(self, x, *args, **kwargs):
        # x = x.reshape(x.shape[0], -1, self.d_input*self.chunk)
        T = x.shape[1]
        h = torch.zeros(x.shape[0], self.d_model, device=x.device)

        for i in range(T):
            # z = self.E(x[:, i, :])
            z = x[:, i, :]

            if self.model == 'simpleRNN':
                h = self.tanh(self.W(h) + z)

            elif self.model == 'resRNN':
                h = h + self.eps * self.tanh(self.W(h) + z)

            elif self.model == 'asymRNN':
                if i == 0:
                    W = self.C - self.C.transpose(1, 0) - self.gamma * self.I
                h = h + self.eps * self.tanh(torch.matmul(h, W) + z)

            elif self.model == 'calRNN':
                if i == 0:
                    C = self.C - self.C.transpose(1, 0)
                    W = torch.matmul(torch.inverse(self.I + C), self.I - C)
                h = self.tanh(torch.matmul(h, W) + z)

            elif self.model == 'LipschitzRNN':
                if i == 0:
                    A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                        1 - self.beta) * (self.B + self.B.transpose(
                            1, 0)) - self.gamma * self.I
                    W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                        1 - self.beta) * (self.C + self.C.transpose(
                            1, 0)) - self.gamma * self.I
                h = h + self.eps * self.alpha * torch.matmul(
                    h, A) + self.eps * self.tanh(torch.matmul(h, W) + z)

            elif self.model == 'LipschitzRNN_gated':
                if i == 0:
                    A = self.beta * (self.B - self.B.transpose(1, 0)) + (
                        1 - self.beta) * (self.B + self.B.transpose(
                            1, 0)) - self.gamma * self.I
                    W = self.beta * (self.C - self.C.transpose(1, 0)) + (
                        1 - self.beta) * (self.C + self.C.transpose(
                            1, 0)) - self.gamma * self.I
                z_gate = self.E_gate(x[:, i, :])
                Wh = torch.matmul(h, W)
                Ah = torch.matmul(h, A)
                q1 = self.alpha * Ah + self.tanh(Wh + z)
                q2 = self.sigmoid(Wh + z_gate)
                h = h + self.eps * q1 * q2

            elif self.model == 'LipschitzRNN_ODE':
                self.func.z = z
                self.func.i = i
                h = odeint(self.func,
                           h,
                           torch.tensor([0, self.eps]).float(),
                           method=self.solver)[-1, :, :]

        # Decoder
        #----------
        # out = self.D(h)
        # return out

        return h.unsqueeze(1), None
