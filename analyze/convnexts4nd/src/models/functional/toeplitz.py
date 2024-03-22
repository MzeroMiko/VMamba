"""Utilities for computing convolutions.

There are 3 equivalent views:
    1. causal convolution
    2. multiplication of (lower) triangular Toeplitz matrices
    3. polynomial multiplication (mod x^N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def construct_toeplitz(v, f=0.0):
    """Explicit construction of Krylov matrix [v  A @ v  A^2 @ v  ...  A^{n-1} @ v]
    where A = Z_f. This uses vectorized indexing and cumprod so it's much
    faster than using the Krylov function.
    Parameters:
        v: the starting vector of size n or (rank, n).
        f: real number
    Returns:
        K: Krylov matrix of size (n, n) or (rank, n, n).
    """
    n  = v.shape[-1]
    a = torch.arange(n, device=v.device)
    b = -a
    indices = a[:, None] + b[None]
    K = v[..., indices]
    K[..., indices < 0] *= f
    return K

def triangular_toeplitz_multiply_(u, v, sum=None):
    n = u.shape[-1]
    u_expand = F.pad(u, (0, n))
    v_expand = F.pad(v, (0, n))
    u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
    v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)
    uv_f = u_f * v_f
    if sum is not None:
        uv_f = uv_f.sum(dim=sum)
    output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
    return output

def triangular_toeplitz_multiply_padded_(u, v):
    """ Same as triangular_toeplitz_multiply but inputs and output assume to be 0-padded already. """
    n = u.shape[-1]
    assert n % 2 == 0
    u_f = torch.fft.rfft(u, n=n, dim=-1)
    v_f = torch.fft.rfft(v, n=n, dim=-1)
    uv_f = u_f * v_f
    output = torch.fft.irfft(uv_f, n=n, dim=-1)
    output[..., n:] = 0
    return output

class TriangularToeplitzMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        return triangular_toeplitz_multiply_(u, v)

    @staticmethod
    def backward(ctx, grad):
        u, v = ctx.saved_tensors
        d_u = triangular_toeplitz_multiply_(grad.flip(-1), v).flip(-1)
        d_v = triangular_toeplitz_multiply_(grad.flip(-1), u).flip(-1)
        return d_u, d_v

class TriangularToeplitzMultFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_expand = F.pad(u, (0, n))
        v_expand = F.pad(v, (0, n))
        u_f = torch.fft.rfft(u_expand, n=2*n, dim=-1)
        v_f = torch.fft.rfft(v_expand, n=2*n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=2*n, dim=-1)[..., :n]
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad.flip(-1), (0, n))
        g_f = torch.fft.rfft(g_expand, n=2*n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=2*n, dim=-1)[..., :n]
        d_v = torch.fft.irfft(gu_f, n=2*n, dim=-1)[..., :n]
        d_u = d_u.flip(-1)
        d_v = d_v.flip(-1)
        return d_u, d_v

class TriangularToeplitzMultPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v):
        ctx.save_for_backward(u, v)
        output = triangular_toeplitz_multiply_(u, v)
        return output

    @staticmethod
    def backward(ctx, grad):
        u, v = ctx.saved_tensors
        d_u = triangular_toeplitz_multiply_padded_(grad.flip(-1), v).flip(-1)
        d_v = triangular_toeplitz_multiply_padded_(grad.flip(-1), u).flip(-1)
        return d_u, d_v

class TriangularToeplitzMultPaddedFast(torch.autograd.Function):
    """ Trade off speed (20-25% faster) for more memory (20-25%) """

    @staticmethod
    def forward(ctx, u, v):
        n = u.shape[-1]
        u_f = torch.fft.rfft(u, n=n, dim=-1)
        v_f = torch.fft.rfft(v, n=n, dim=-1)

        ctx.save_for_backward(u_f, v_f)

        uv_f = u_f * v_f
        output = torch.fft.irfft(uv_f, n=n, dim=-1)
        output[..., n//2:].zero_()
        return output

    @staticmethod
    def backward(ctx, grad):
        u_f, v_f = ctx.saved_tensors
        n = grad.shape[-1]
        g_expand = F.pad(grad[..., :n//2].flip(-1), (0, n//2))
        g_f = torch.fft.rfft(g_expand, n=n, dim=-1)
        gu_f = g_f * u_f
        gv_f = g_f * v_f
        d_u = torch.fft.irfft(gv_f, n=n, dim=-1)
        d_v = torch.fft.irfft(gu_f, n=n, dim=-1)
        d_u[..., n//2:].zero_()
        d_v[..., n//2:].zero_()
        d_u[..., :n//2] = d_u[..., :n//2].flip(-1) # TODO
        d_v[..., :n//2] = d_v[..., :n//2].flip(-1) # TODO
        return d_u, d_v

# triangular_toeplitz_multiply = triangular_toeplitz_multiply_
triangular_toeplitz_multiply = TriangularToeplitzMult.apply
triangular_toeplitz_multiply_fast = TriangularToeplitzMultFast.apply
triangular_toeplitz_multiply_padded = TriangularToeplitzMultPadded.apply
triangular_toeplitz_multiply_padded_fast = TriangularToeplitzMultPaddedFast.apply

def causal_convolution(u, v, fast=True, pad=False):
    if not pad and not fast:
        return triangular_toeplitz_multiply(u, v)
    if not pad and fast:
        return triangular_toeplitz_multiply_fast(u, v)
    if pad and not fast:
        return triangular_toeplitz_multiply_padded(u, v)
    if pad and fast:
        return triangular_toeplitz_multiply_padded_fast(u, v)

def _fft(x, N): return torch.fft.rfft(F.pad(x, (0, 2*N-x.shape[-1])), n=2*N, dim=-1)
def _ifft(x, N): return torch.fft.irfft(x, n=2*N, dim=-1)[..., :N]

def causal_convolution_inverse(u):
    """ Invert the causal convolution/polynomial/triangular Toeplitz matrix represented by u.

    This is easiest in the polynomial view:
    https://www.csa.iisc.ac.in/~chandan/courses/CNT/notes/lec5.pdf
    The idea is that
    h = g^{-1} (mod x^m) => 2h - gh^2 = g^{-1} (mod x^{2m})

    # TODO this can be numerically unstable if input is "poorly conditioned",
    # for example if u[0] is magnitudes different from the rest of u
    """
    N = u.shape[-1]
    v = u[..., :1].reciprocal()
    while v.shape[-1] < N:
        M = v.shape[-1]
        v_f = _fft(v, 2*M)
        u_f = _fft(u[..., :2*M], 2*M)
        _v = -_ifft(u_f * v_f**2, 2*M)
        _v[..., :M] = _v[..., :M] + 2*v
        v = _v
    # TODO contiguous?
    v = v[..., :N]
    return v
