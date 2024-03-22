"""pykeops implementations of the core Cauchy kernel used in the S4 algorithm.

The interface of the Cauchy multiplication is:

Inputs:
  v: (N)
  z: (N)
  w: (L)

Returns: y (L)
  y_k = \sum_i v_i / (z_i - w_k)
"""

import torch
from einops import rearrange

try:
    import pykeops
    from pykeops.torch import LazyTensor, Genred
except:
    pass

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors

def _c2r(x): return torch.view_as_real(x)
def _r2c(x): return torch.view_as_complex(x)

def cauchy_naive(v, z, w, conj=True):
    """
    v: (..., N)
    z: (..., L)
    w: (..., N)
    returns: (..., L) \sum v/(z-w)
    """
    if conj:
        v = _conj(v)
        w = _conj(w)
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1)) # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)

def cauchy_lazy(v, z, w, conj=True):
    if conj:
        v = _conj(v)
        w = _conj(w)
    v, z, w = _broadcast_dims(v, z, w)
    v_l = LazyTensor(rearrange(v, '... N -> ... N 1 1'))
    w_l = LazyTensor(rearrange(w, '... N -> ... N 1 1'))
    z_l = LazyTensor(rearrange(z, '... L -> ... 1 L 1'))
    sub = z_l - w_l  # (b N L 1), for some reason it doesn't display the last dimension
    div = v_l / sub
    s = div.sum(dim=len(v_l.shape)-2)
    return s.squeeze(-1)

def cauchy(v, z, w, conj=False):
    expr = 'ComplexDivide(v, z-w)'
    cauchy_mult = Genred(
        expr,
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    if conj:
        v = _conj(v)
        w = _conj(w)
    v, z, w = _broadcast_dims(v, z, w)
    v = _c2r(v)
    z = _c2r(z)
    w = _c2r(w)

    r = cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)

def cauchy_real(v, z, w):
    expr = 'v / (z - w)'
    cauchy_mult = Genred(
        expr,
        [
            'v = Vj(1)',
            'z = Vi(1)',
            'w = Vj(1)',
        ],
        reduction_op='Sum',
        axis=1,
    )
    v, z, w = _broadcast_dims(v, z, w)
    v = v.unsqueeze(-1)
    z = z.unsqueeze(-1)
    w = w.unsqueeze(-1)

    r = cauchy_mult(v, z, w, backend='GPU')
    return r


def cauchy_conj(v, z, w, num=2, denom=2):
    if num == 1:
        expr_num = 'z * ComplexReal(v) - Real2Complex(ComplexReal(v)*ComplexReal(w) + ComplexImag(v)*ComplexImag(w))'
    elif num == 2:
        expr_num = 'z * ComplexReal(v) - Real2Complex(Sum(v * w))'
    else: raise NotImplementedError

    if denom == 1:
        expr_denom = 'ComplexMult(z-Real2Complex(ComplexReal(w)), z-Real2Complex(ComplexReal(w))) + Real2Complex(Square(ComplexImag(w)))'
    elif denom == 2:
        expr_denom = 'ComplexMult(z-w, z-Conj(w))'
    else: raise NotImplementedError

    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = _c2r(v)
    z = _c2r(z)
    w = _c2r(w)

    r = 2*cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)

def cauchy_conj_components(v, z, w):
    """ Assumes z is pure imaginary (as in S4 with bilinear) """

    expr_num = 'Imag2Complex(zi*vr) - Real2Complex(vr*wr + vi*wi)'
    expr_denom = 'Real2Complex(Square(wr)+Square(wi)-Square(zi)) - Imag2Complex(IntCst(2)*zi*wr)'
    cauchy_mult = Genred(
        f'ComplexDivide({expr_num}, {expr_denom})',
        [
            'vr = Vj(1)',
            'vi = Vj(1)',
            'wr = Vj(1)',
            'wi = Vj(1)',
            'zi = Vi(1)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    v, z, w = _broadcast_dims(v, z, w)
    v = v.unsqueeze(-1)
    z = z.unsqueeze(-1)
    w = w.unsqueeze(-1)

    v_r, v_i = v.real.contiguous(), v.imag.contiguous()
    w_r, w_i = w.real.contiguous(), w.imag.contiguous()
    z_i = z.imag.contiguous()

    r = 2*cauchy_mult(v_r, v_i, w_r, w_i, z_i, backend='GPU')
    return _r2c(r)

def cauchy_conj_components_lazy(v, z, w, type=1):
    v, z, w = _broadcast_dims(v, z, w)

    v_r, v_i = v.real.contiguous(), v.imag.contiguous()
    w_r, w_i = w.real.contiguous(), w.imag.contiguous()
    z_i = z.imag.contiguous()

    v_r = LazyTensor(rearrange(v_r, '... N -> ... 1 N 1'))
    v_i = LazyTensor(rearrange(v_i, '... N -> ... 1 N 1'))
    w_r = LazyTensor(rearrange(w_r, '... N -> ... 1 N 1'))
    w_i = LazyTensor(rearrange(w_i, '... N -> ... 1 N 1'))
    z_i = LazyTensor(rearrange(z_i, '... L -> ... L 1 1'))

    if type == 1:
        num = -v_r*w_r-v_i*w_i + 1j* z_i*v_r
        denom = w_r**2+w_i**2-z_i**2 - 2j*w_r*z_i
    else:
        # z = torch.complex(-w_r, z_i) # Not supported
        z = -w_r + 1j* z_i
        num = v_r * z - v_i*w_i
        denom = z*z + w_i**2 # z**2 is bugged for complex

    r = num / denom
    r = 2*r.sum(dim=len(z_i.shape)-1)
    return r.squeeze(-1)

def cauchy_conj2(v, z, w):
    expr = 'ComplexDivide(v, z-w) + ComplexDivide(Conj(v), z-Conj(w))'
    cauchy_mult = Genred(
        expr,
        [
            'v = Vj(2)',
            'z = Vi(2)',
            'w = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    v, z, w = _broadcast_dims(v, z, w)
    if complex:
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

    r = cauchy_mult(v, z, w, backend='GPU')
    return _r2c(r)
