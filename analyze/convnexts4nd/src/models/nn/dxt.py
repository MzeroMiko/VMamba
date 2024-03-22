"""Implementations of several types of Discrete Sin/Cosine Transforms with various reductions to FFT.

Currently not used by S4.
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.fft
from einops import rearrange, repeat

class DCT(nn.Module):
    """Reductions adapted from https://dsp.stackexchange.com/questions/2807/fast-cosine-transform-via-fft."""

    def __init__(self, N, norm='backward'):
        super().__init__()

        self.N = N

        # Materialize DCT matrix
        P = scipy.fft.dct(np.eye(N), norm=norm, type=2).T
        P = torch.tensor(P, dtype=torch.float)
        self.register_buffer('P', P)

        # TODO take care of normalization
        Q = np.exp(-1j * np.pi / (2 * self.N) * np.arange(self.N))
        Q = torch.tensor(Q, dtype=torch.cfloat)
        self.register_buffer('Q', Q) # half shift

    def forward(self, x, mode=2):
        if mode == 0:
            return self.forward_dense(x)
        elif mode == 1:
            return self.forward_n(x)
        elif mode == 2:
            return self.forward_2n(x)
        elif mode == 4:
            return self.forward_4n(x)

    def forward_dense(self, x):
        """Baseline DCT type II - matmul by DCT matrix."""
        y = (self.P.to(x) @ x.unsqueeze(-1)).squeeze(-1)
        return y

    def forward_4n(self, x):
        """DCT type II - reduction to FFT size 4N."""
        assert self.N == x.shape[-1]
        x = torch.cat([x, x.flip(-1)], dim=-1)
        z = torch.zeros_like(x)
        x = torch.stack([z, x], dim=-1)
        x = x.view(x.shape[:-2] + (-1,))
        y = torch.fft.fft(x)
        y = y[..., :self.N]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_2n(self, x):
        """DCT type II - reduction to FFT size 2N mirrored.

        The reduction from the DSP forum is not quite correct in the complex input case.
        halfshift(FFT[a, b, c, d, d, c, b, a]) -> [A, B, C, D, 0, -D, -C, -B]
        In the case of real input, the intermediate step after FFT has form [A, B, C, D, 0, D*, C*, B*]
        """
        assert self.N == x.shape[-1]
        x = torch.cat([x, x.flip(-1)], dim=-1)
        y = torch.fft.fft(x)[..., :self.N]
        y = y * self.Q
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_n(self, x):
        """ DCT type II - reduction to size N """
        assert self.N == x.shape[-1]
        x = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
        y = torch.fft.fft(x)
        y = y * 2 * self.Q
        if torch.is_complex(x):
            y = torch.cat([y[..., :1], (y[..., 1:] + 1j * y[..., 1:].flip(-1)) / 2], dim=-1) # TODO in-place sum
        else:
            y = torch.real(y)
        return y

class IDCT(nn.Module):
    def __init__(self, N, norm='backward'):
        super().__init__()

        self.N = N

        # Materialize DCT matrix
        P = np.linalg.inv(scipy.fft.dct(np.eye(N), norm=norm, type=2).T)
        P = torch.tensor(P, dtype=torch.float)
        self.register_buffer('P', P)

        # TODO take care of normalization
        Q = np.exp(-1j * np.pi / (2 * self.N) * np.arange(2*self.N))
        Q = torch.tensor(Q, dtype=torch.cfloat)
        self.register_buffer('Q', Q) # half shift

    def forward(self, x, mode=2):
        if mode == 0:
            return self.forward_dense(x)
        elif mode == 1:
            return self.forward_n(x)
        elif mode == 2:
            return self.forward_2n(x)
        elif mode == 4:
            return self.forward_4n(x)

    def forward_dense(self, x):
        """Baseline DCT type II - matmul by DCT matrix."""
        y = (self.P.to(x) @ x.unsqueeze(-1)).squeeze(-1)
        return y

    def forward_4n(self, x):
        """DCT type II - reduction to FFT size 4N."""
        assert self.N == x.shape[-1]
        z = x.new_zeros(x.shape[:-1] + (1,))
        x = torch.cat([x, z, -x.flip(-1), -x[..., 1:], z, x[..., 1:].flip(-1)], dim=-1)
        y = torch.fft.ifft(x)
        y = y[..., 1:2*self.N:2]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_2n(self, x):
        """DCT type II - reduction to FFT size 2N mirrored."""
        assert self.N == x.shape[-1]
        z = x.new_zeros(x.shape[:-1] + (1,))
        x = torch.cat([x, z, -x[..., 1:].flip(-1)], dim=-1)
        x = x / self.Q
        y = torch.fft.ifft(x)[..., :self.N]
        if torch.is_complex(x):
            return y
        else:
            return torch.real(y)

    def forward_n(self, x):
        """DCT type II - reduction to size N."""
        assert self.N == x.shape[-1]
        raise NotImplementedError # Straightforward by inverting operations of DCT-II reduction
