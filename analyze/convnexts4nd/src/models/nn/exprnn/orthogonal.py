# Adapted from https://github.com/Lezcano/expRNN

import torch
import torch.nn as nn

from .parametrization import Parametrization
from src.models.nn.activation import ModReLU


class Orthogonal(Parametrization):
    """Class that implements optimization restricted to the Stiefel manifold."""

    def __init__(self, d_input, d_output, initializer_skew, mode, param):
        """
        mode: "static" or a tuple such that:
                mode[0] == "dynamic"
                mode[1]: int, K, the number of steps after which we should change the basis of the dyn triv
                mode[2]: int, M, the number of changes of basis after which we should project back onto the manifold the basis. This is particularly helpful for small values of K.

        param: A parametrization of in terms of skew-symmetyric matrices
        """
        max_size = max(d_input, d_output)
        A = torch.empty(max_size, max_size)
        base = torch.empty(d_input, d_output)
        super(Orthogonal, self).__init__(A, base, mode)
        self.d_input = d_input
        self.d_output = d_output
        self.param = param
        self.init_A = initializer_skew
        self.init_base = nn.init.eye_

        self.reset_parameters()

    def reset_parameters(self):
        self.init_A(self.A)
        self.init_base(self.base)

    def forward(self, input):
        return input.matmul(self.B)

    def retraction(self, A, base):
        # This could be any parametrization of a tangent space
        A = A.triu(diagonal=1)
        A = A - A.t()
        B = base.mm(self.param(A))
        if self.d_input != self.d_output:
            B = B[:self.d_input, :self.d_output]
        return B

    def project(self, base):
        try:
            # Compute the projection using the thin SVD decomposition
            U, _, V = torch.svd(base, some=True)
            return U.mm(V.t())
        except RuntimeError:
            # If the svd does not converge, fallback to the (thin) QR decomposition
            x = base
            if base.size(0) < base.size(1):
                x = base.t()
            ret = torch.qr(x, some=True).Q
            if base.size(0) < base.size(1):
                ret = ret.t()
            return ret


class OrthogonalRNN(nn.Module):
    def __init__(self, d_input, d_model, initializer_skew, mode, param):
        super(OrthogonalRNN, self).__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.recurrent_kernel = Orthogonal(d_model, d_model, initializer_skew, mode, param=param)
        self.input_kernel = nn.Linear(in_features=self.d_input, out_features=self.d_model, bias=False)
        self.nonlinearity = ModReLU(d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.input_kernel.weight.data, nonlinearity="relu")

    def default_hidden(self, input):
        return input.new_zeros(input.size(0), self.d_model, requires_grad=False)

    def forward(self, input, hidden):
        input = self.input_kernel(input)
        hidden = self.recurrent_kernel(hidden)
        out = input + hidden
        out = self.nonlinearity(out)

        return out, out
