"""Defines flexible gating mechanisms.

Based on ideas from LSSL paper and UR-LSTM paper (https://arxiv.org/abs/1910.09890).
"""

import torch
import torch.nn as nn

class Gate(nn.Module):
    """Implements gating mechanisms.

    LSSL paper elaborates on the most import connection: A standard sigmoid gate
    is equivalent to an exponential parameterization + Backwards Euler disc.

    Mechanisms:
    N  - No gate
    G  - Standard sigmoid gate
    UR - Uniform refine gates
    R  - Refine gate

    FS - Forward discretization, Sigmoid activation [equivalent to G]
    BE - Backward discretization, Exp activation [equivalent to G]
    BR - Backward discretization, Relu activation
    TE - Trapezoid discretization, Exp activation
    TR - Trapezoid discretization, Relu activation
    TS - Trapezoid discretization, Sigmoid activation (0 to 2)
    """

    def __init__(self, size, preact_ctor, preact_args, mechanism='N'):
        super().__init__()
        self.size      = size
        self.mechanism = mechanism

        if self.mechanism == 'N':
            pass
        elif self.mechanism in ['G', 'FS', 'BE', 'BR', 'TE', 'TR', 'TS', 'ZE', 'ZR', 'ZS']:
            self.W_g = preact_ctor(*preact_args)
        elif self.mechanism in ['U', 'UT']:
            self.W_g = preact_ctor(*preact_args)
            b_g_unif = torch.empty(size)
            torch.nn.init.uniform_(b_g_unif, 1./self.size, 1.-1./self.size)
            self.b_g = nn.Parameter(torch.log(1./b_g_unif-1.).detach(), requires_grad=False)
        elif self.mechanism == 'UR':
            self.W_g = preact_ctor(*preact_args)
            self.W_r = preact_ctor(*preact_args)

            b_g_unif = torch.empty(size)
            torch.nn.init.uniform_(b_g_unif, 1./self.size, 1.-1./self.size)
            self.b_g = nn.Parameter(torch.log(1./b_g_unif-1.).detach(), requires_grad=False)
        elif self.mechanism == 'R':
            self.W_g = preact_ctor(*preact_args)
            self.W_r = preact_ctor(*preact_args)
        elif self.mechanism in ['GT']:
            self.W_g = preact_ctor(*preact_args)
        else:
            assert False, f'Gating type {self.mechanism} is not supported.'

    def forward(self, *inputs):
        if self.mechanism == 'N':
            return 1.0

        if self.mechanism == 'G':
            g_preact = self.W_g(*inputs)
            g = torch.sigmoid(g_preact)
        if self.mechanism == 'U':
            g_preact = self.W_g(*inputs) + self.b_g
            g = torch.sigmoid(g_preact)
        elif self.mechanism == 'UR':
            g_preact = self.W_g(*inputs) + self.b_g
            g = torch.sigmoid(g_preact)
            r = torch.sigmoid(self.W_r(*inputs))
            g = (1-2*r)*g**2 + 2*r*g
        elif self.mechanism == 'R':
            g_preact = self.W_g(*inputs)
            g = torch.sigmoid(g_preact)
            r = torch.sigmoid(self.W_r(*inputs))
            g = (1-2*r)*g**2 + 2*r*g
        elif self.mechanism == 'UT':
            g_preact = self.W_g(*inputs) + self.b_g
            g = torch.sigmoid(g_preact)
            r = g
            g = (1-2*r)*g**2 + 2*r*g
        elif self.mechanism == 'GT':
            g_preact = self.W_g(*inputs)
            g = torch.sigmoid(g_preact)
            r = g
            g = (1-2*r)*g**2 + 2*r*g
        else:
            g_preact = self.W_g(*inputs)
            # if self.mechanism[1] == 'S':
            #     g = torch.sigmoid(g_preact)
            # elif self.mechanism[1] == 'E':
            #     g = torch.exp(g_preact)
            # elif self.mechanism[1] == 'R':
            #     g = torch.relu(g_preact)
            if self.mechanism == 'FS':
                g = torch.sigmoid(g_preact)
                g = self.forward_diff(g)
            elif self.mechanism == 'BE':
                g = torch.exp(g_preact)
                g = self.backward_diff(g)
            elif self.mechanism == 'BR':
                g = torch.relu(g_preact)
                g = self.backward_diff(g)
            elif self.mechanism == 'TS':
                g = 2 * torch.sigmoid(g_preact)
                g = self.trapezoid(g)
            elif self.mechanism == 'TE':
                g = torch.exp(g_preact)
                g = self.trapezoid(g)
            elif self.mechanism == 'TR':
                g = torch.relu(g_preact)
                g = self.trapezoid(g)
            elif self.mechanism == 'ZE':
                g = torch.exp(g_preact)
                g = self.zoh(g)
            elif self.mechanism == 'ZR':
                g = torch.relu(g_preact)
                g = self.zoh(g)
            elif self.mechanism == 'ZS':
                g = torch.sigmoid(g_preact)
                g = self.zoh(g)
        return g

    def forward_diff(self, x):
        return x

    def backward_diff(self, x):
        return x / (1+x)

    def trapezoid(self, x):
        return x / (1 + x/2)

    def zoh(self, x):
        return 1 - torch.exp(-x)
