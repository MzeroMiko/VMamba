"""Utilities to calculate the transitions of the HiPPO ODE x' = Ax + Bu and discrete-time recurrence approximation.

Note that these modules were heavily used in LSSL, but is no longed needed for S4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import special as ss
from einops import rearrange

from src.models.hippo.hippo import transition
from src.models.functional.toeplitz import causal_convolution, causal_convolution_inverse, construct_toeplitz

# TODO figure out if we actually need this
try:
    from extensions.legt.legt import legt_gbt_forward, legt_gbt_backward, legt_gbt_forward_t, legt_gbt_backward_t
except:
    pass

try:
    from extensions.trid.trid import trid_gbt_forward, trid_gbt_backward, trid_solve
except:
    pass
# from pytorch_memlab import profile


class AdaptiveTransition(nn.Module):
    def __init__(self, N, params, trainable=False, lr=1.0, batch=()):
        """
        params: dict of Tensors that encode the parameters of the state system A, B.
        """

        super().__init__()
        self.N = N
        self.trainable = trainable
        self.batch = batch

        if self.trainable:
            for name, p in params.items():
                p = p.repeat(*batch, *[1]*len(p.shape))
                self.register_parameter(name, nn.Parameter(p))
                getattr(self, name)._lr = lr
        else:
            assert batch == (), "If not learnable, Transition should not have a batch dimension"
            for name, p in params.items():
                self.register_buffer(name, p)

        # Register some common buffers
        # (helps make sure every subclass has access to them on the right device)
        I = torch.eye(N)
        self.register_buffer('I', I)
        self.register_buffer('ones', torch.ones(N))
        self.register_buffer('arange', torch.arange(N))


    @property
    def A(self):
        if self.trainable:
            return self._A()
        # Cache it the first time this is called
        # this must be done here and not in __init__ so all tensors are on the right device
        else:
            if not hasattr(self, '_cached_A'):
                self._cached_A = self._A()
            return self._cached_A

    @property
    def B(self):
        if self.trainable:
            return self._B()
        # Cache it the first time this is called
        # this must be done here and not in __init__ so all tensors are on the right device
        else:
            if not hasattr(self, '_cached_B'):
                self._cached_B = self._B()
            return self._cached_B

    def precompute_forward(self):
        raise NotImplementedError

    def precompute_backward(self):
        raise NotImplementedError

    def forward_mult(self, u, delta):
        """Computes (I + delta A) u.

        A: (n, n)
        u: (..., n)
        delta: (...) or scalar

        output: (..., n)
        """
        raise NotImplementedError

    def inverse_mult(self, u, delta): # TODO swap u, delta everywhere
        """Computes (I - d A)^-1 u."""
        raise NotImplementedError

    def forward_diff(self, d, u, v):
        """Computes the 'forward diff' or Euler update rule: (I - d A)^-1 u + d B v.

        d: (...)
        u: (..., n)
        v: (...)
        """

        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = self.forward_mult(u, d)
        x = x + v
        return x

    def backward_diff(self, d, u, v):
        """Computes the 'forward diff' or Euler update rule: (I - d A)^-1 u + d (I - d A)^-1 B v.

        d: (...)
        u: (..., n)
        v: (...)
        """

        v = d * v
        v = v.unsqueeze(-1) * self.B
        x = u + v
        x = self.inverse_mult(x, d)
        return x

    def bilinear(self, dt, u, v, alpha=.5):
        """Computes the bilinear (aka trapezoid or Tustin's) update rule.

        (I - d/2 A)^-1 (I + d/2 A) u + d B (I - d/2 A)^-1 B v

        dt: (...)
        u: (..., N)
        v: (...)
        """

        x = self.forward_mult(u, (1-alpha)*dt)
        v = dt * v
        v = v.unsqueeze(-1) * self.B
        x = x + v
        x = self.inverse_mult(x, (alpha)*dt)
        return x

    def zoh(self, dt, u, v):
        raise NotImplementedError

    def gbt_A(self, dt, alpha=.5):
        """Compute the transition matrices associated with bilinear transform.

        dt: (...) broadcastable with self.batch_shape
        returns: (..., N, N)
        """

        # Solve (N, ...) parallel problems of size N
        dims = max(len(dt.shape), len(self.batch))
        I = self.I.view([self.N] + [1]*dims + [self.N])
        A = self.bilinear(dt, I, dt.new_zeros(*dt.shape), alpha=alpha) # (N, ..., N)
        A = rearrange(A, 'n ... m -> ... m n', n=self.N, m=self.N)
        return A

    def gbt_B(self, dt, alpha=.5):
        B = self.bilinear(dt, dt.new_zeros(*dt.shape, self.N), dt.new_ones(1), alpha=alpha) # (..., N)
        return B

class ManualAdaptiveTransition(AdaptiveTransition):
    def __init__(self, N, A, B, **kwargs):
        """
        A: (N, N)
        B: (N,)
        """

        super().__init__(N, {'a': A, 'b': B}, **kwargs)

    def _A(self):
        return self.a

    def _B(self):
        return self.b

    # TODO necessary?
    def precompute_forward(self, delta):
        return self.I + delta*self.A

    def precompute_backward(self, delta):
        return torch.linalg.solve(self.I - delta*self.A, self.I)[0]


    def quadratic(self, x, y):
        """Implements the quadratic form given by the A matrix.
        x : (..., N)
        y : (..., N)
        returns: x^T A y (...)
        """

        return torch.sum((self.A @ y.unsqueeze(-1)).squeeze(-1) * x, dim=-1)

    def forward_mult(self, u, delta, transpose=False):
        """Computes (I + d A) u.

        A: (n, n)
        u: (b1* d, n) d represents memory_size
        delta: (b2*, d) or scalar
          Assume len(b2) <= len(b1)

        output: (broadcast(b1, b2)*, d, n)
        """

        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)
        A_ = self.A.transpose(-1, -2) if transpose else self.A
        x = (A_ @ u.unsqueeze(-1)).squeeze(-1)
        x = u + delta * x

        return x


    def inverse_mult(self, u, delta, transpose=False):
        """Computes (I - d A)^-1 u."""

        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1).unsqueeze(-1)
        _A = self.I - delta * self.A
        if transpose: _A = _A.transpose(-1, -2)

        # x = torch.linalg.solve(_A, u.unsqueeze(-1)).squeeze(-1)

        # TODO pass in a flag to toggle the two codepaths depending on how big the problem is
        xs = []
        for _A_, u_ in zip(*torch.broadcast_tensors(_A, u.unsqueeze(-1))):
            x_ = torch.linalg.solve(_A_, u_[...,:1]).squeeze(-1)
            xs.append(x_)
        x = torch.stack(xs, dim=0)

        return x


class OPManualAdaptiveTransition(ManualAdaptiveTransition):
    measure = None

    def __init__(self, N, verbose=False, measure_args={}, **kwargs):
        """ Slow (n^3, or n^2 if step sizes are cached) version via manual matrix mult/inv

        delta: optional list of step sizes to cache the transitions for
        """
        A, B = transition(type(self).measure, N, **measure_args)
        A = torch.as_tensor(A, dtype=torch.float)
        B = torch.as_tensor(B, dtype=torch.float)[:, 0]
        super().__init__(N, A, B, **kwargs)


        if verbose:
            print(f"{self.__class__}\n  A {self.A}\nB {self.B}")


class LegSAdaptiveTransitionManual(OPManualAdaptiveTransition):
    measure = 'legs'

class LegTAdaptiveTransitionManual(OPManualAdaptiveTransition):
    measure = 'legt'

class LagTAdaptiveTransitionManual(OPManualAdaptiveTransition):
    measure = 'lagt'

class TLagTAdaptiveTransitionManual(OPManualAdaptiveTransition):
    measure = 'tlagt'

class GLagTAdaptiveTransitionManual(OPManualAdaptiveTransition):
    measure = 'glagt'


# TODO this class is not learnable for now (will have to change the shape of a, b to [1])
class CumsumAdaptiveTransition(AdaptiveTransition):
    def __init__(self, N, a, b):
        """Implements update for matrix A = -(L+aI) for forward, backward, bilinear, zoh discretizations.
        a: scalar, the element on the diagonal
        b: scalar, so that B = b * ones vector
        """

        # Can't wrap scalars with torch.Tensor(), while torch.tensor(a) gives double instead of float or something
        # super().__init__(N, {'a': [a], 'b': [b]}, **kwargs) # TODO this should register b and then construct self.B using a @property, like in Toeplitz (but is slightly slower in the non-learnable case)
        params = {
            'a': torch.tensor(a, dtype=torch.float),
            'b': torch.tensor(b, dtype=torch.float),
        }
        super().__init__(N, params)

        # self.N = N
        # self.a = a
        # self.b = b

        # self.register_buffer('A', self.construct_A())
        # self.register_buffer('B', b * torch.ones(N))

        # self.register_buffer('I', torch.eye(N))
        self.register_buffer('arange', torch.arange(N-1))


    def _A(self):
        L = torch.tril(self.ones.repeat(self.N, 1))
        D = self.a * self.I
        return -(L+D)

    def _B(self):
        return self.b * self.ones


    def quadratic(self, x, y):
        """
        x : (..., N)
        y : (..., N)
        returns: x^T A y (...)
        """
        return torch.sum((self.A @ y.unsqueeze(-1)).squeeze(-1) * x, dim=-1)

    def precompute_forward(self, delta):
        """Store elements along the diagonals of (I + d A)."""

        if isinstance(delta, float):
            delta = torch.tensor(delta).to(self.I)
        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)
        a_ = 1. - delta * self.a # (..., 1)
        if self.N == 1:
            return a_
        return torch.cat((a_, -delta*delta.new_ones(self.N-1)), -1) # (..., N)

    def precompute_backward(self, delta): # TODO should be called inverse?
        """Store elements along the diagonals of (I - d A)^{-1}.

        # a' = a + 1/dt
        delta: (...)
        output: (..., N)
        """

        if isinstance(delta, float):
            delta = torch.tensor(delta).to(self.I)
        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)

        if self.N == 1:
            return 1. / (1. + self.a*delta + delta)

        ad = self.a*delta # (..., 1)
        ad_p1 = 1 + ad
        denom = ad_p1 + delta # 1 + a'
        denom_inv = denom.reciprocal() # 1. / denom
        s = - delta * denom_inv * denom_inv # -1/(1+a')^2
        b = ad_p1 * denom_inv # a' / (1 + a')
        pows = b ** self.arange ## TODO benchmark against cumprod or cumsum in log space
        tail = s * pows
        ret = torch.cat((denom_inv, tail), -1)
        return ret
        # ad = self.a*delta # (..., 1)
        # denom = 1 + ad + delta
        # s = - delta / denom# -1/(1+a')
        # b = (1 + ad) / denom # a' / (1 + a')
        # # pows = b ** torch.arange(self.N-1).to(self.I) ## TODO benchmark against cumprod or cumsum in log space
        # pows = b ** self.arange ## TODO benchmark against cumprod or cumsum in log space
        # tail = s * pows
        # ret = torch.cat((tail.new_ones(tail.shape[:-1]+(1,)), tail), -1)
        # ret = ret / denom
        # return ret

    def precompute_gbt_A(self, delta, alpha=0.5):
        """Return the A matrix of the gbt discretization."""
        c = self.precompute_forward((1.-alpha)*delta)
        d = self.precompute_backward(alpha*delta)
        return causal_convolution(c, d)

    def precompute_gbt_B(self, delta, alpha=0.5):
        """Return the B matrix of the gbt discretization."""
        d = self.precompute_backward(alpha*delta)
        # return causal_convolution(d, torch.ones_like(d)) * self.b
        return torch.cumsum(d, -1) * self.b

    def forward_mult(self, u, delta, transpose=False):
        """Computes (I + delta A) u.

        A: (n, n)
        u: (..., n)
        delta: (...) or scalar

        output: (..., n)
        """

        if isinstance(delta, torch.Tensor):
            delta = delta.unsqueeze(-1)

        if transpose:
            x = torch.cumsum(u.flip(-1), -1).flip(-1)
        else:
            x = torch.cumsum(u, -1)
        x = x + u * self.a
        x = u - delta * x # Because A is negated in the representation
        return x

    def inverse_mult(self, u, delta, transpose=False):
        """Computes (I - d A)^-1 u."""
        # if isinstance(delta, torch.Tensor):
        #     delta = delta.unsqueeze(-1)
        # if isinstance(delta, float) and delta in self.backward_cache:
        #     c = self.backward_cache[delta]
        # else:
        # c = self.precompute_backward(delta, **kwargs)
        c = self.precompute_backward(delta)
        if transpose:
            x = causal_convolution(c, u.flip(-1)).flip(-1)
        else:
            x = causal_convolution(c, u)
        return x

class LagTCumsumAdaptiveTransition(CumsumAdaptiveTransition):
    measure = 'lagt'
    def __init__(self, N, beta=1.0):
        # super().__init__(N, -0.5, 1.0)
        super().__init__(N, -0.5, beta)

        # print(f"LagTCumsumAdaptiveTransition:\n  A {self.A}\nB {self.B}")

class TLagTCumsumAdaptiveTransition(CumsumAdaptiveTransition):
    measure = 'tlagt'
    def __init__(self, N, beta=1.0):
        super().__init__(N, -(1.-beta)/2, beta)

        # print(f"LagTCumsumAdaptiveTransition:\n  A {self.A}\nB {self.B}")


class GLagTCumsumAdaptiveTransition(CumsumAdaptiveTransition):
    measure = 'glagt'
    def __init__(self, N, alpha=0.0, beta=0.01):
        # TODO this is completely broken
        raise NotImplementedError
        # super().__init__(N, -(1.-beta)/2, beta)

        # print(f"GLagTCumsumAdaptiveTransition:\n  A {self.A}\nB {self.B}")


class LegTAdaptiveTransition(AdaptiveTransition):
    def __init__(self, N): # this class is not trainable
        A, B = transition('legt', N)
        A = torch.as_tensor(A, dtype=torch.float)
        B = torch.as_tensor(B, dtype=torch.float)[:, 0]
        super().__init__(N, {'a': A, 'b': B})

    def _A(self):
        return self.a

    def _B(self):
        return self.b

    def forward_mult(self, u, delta, transpose=False):
        if transpose: return legt_gbt_forward_t(delta, u, transpose=True) # TODO this is all broken
        else: return legt_gbt_forward(delta, u)

    def inverse_mult(self, u, delta, transpose=False):
        if transpose: return legt_gbt_backward_t(-delta, u, transpose=True)
        else: return legt_gbt_backward(-delta, u)

    def quadratic(self, x, y):
        """
        x : (..., N)
        y : (..., N)
        returns: x^T A y (...)
        """

        # TODO Should use fast mult... also check if we even need this anymore
        return torch.sum((self.A @ y.unsqueeze(-1)).squeeze(-1) * x, dim=-1)

class TriDInverseAdaptiveTransition(AdaptiveTransition):
    # NOTE stores matrix for x' = -Ax + Bu instead of x' = Ax + Bu.

    def __init__(self, N, dl, d, du, pl, pr, c, b, **kwargs):
        params = {
            'dl': dl,
            'd': d,
            'du': du,
            'pl': pl,
            'pr': pr,
            'c': c,
            'b': b,
        }
        super().__init__(N, params, **kwargs)

    def _A(self):
        """The matrix A for system x' = -Ax + Bu."""

        A = trid_solve(self.I, self.dl, self.d, self.du).transpose(-1, -2)
        A = A + self.c*self.I
        A = self.pl.unsqueeze(-1) * A * self.pr
        return A

    def _B(self):
        return self.pl * self.b

    def forward_mult(self, u, delta, transpose=False):
        du = self.du
        d = self.d
        dl = self.dl
        pr = self.pr
        pl = self.pl
        c = self.c
        if transpose:
            return trid_gbt_forward(
                delta, u,
                du, d, dl, pr, pl, c,
            )
        else:
            return trid_gbt_forward(
                delta, u,
                dl, d, du, pl, pr, c,
            )

    def inverse_mult(self, u, delta, transpose=False):
        du = self.du
        d = self.d
        dl = self.dl
        pr = self.pr
        pl = self.pl
        c = self.c
        if transpose:
            return trid_gbt_backward(
                delta, u,
                du, d, dl, pr, pl, c,
            )
        else:
            return trid_gbt_backward(
                delta, u,
                dl, d, du, pl, pr, c,
            )

# TODO turn this into class method
def _diag(N, c): return F.pad(torch.ones(N-1), (1, 1)) * c

class LegTTriDInverseAdaptiveTransition(TriDInverseAdaptiveTransition):
    def __init__(self, N, corners=3, **kwargs):
        p = torch.sqrt(1+2*torch.arange(N))
        # p = torch.ones(N)
        dl = _diag(N, -.5) # + F.pad(torch.randn(N-1)*1e-4, (1, 1))
        du = _diag(N, .5) # + F.pad(torch.randn(N-1)*1e-4, (1, 1))
        d = torch.zeros(N) + torch.randn(N)*1e-2
        if corners == 0:
            pass
        elif corners == 1:
            d[0] += .5
        elif corners == 2:
            d[-1] += .5
        elif corners == 3:
            d[0] += .5
            d[-1] += .5
        else: raise NotImplementedError
        c = torch.ones(N) * 0. # + torch.randn(N)*1e-4

        super().__init__(N, dl, d, du, p, p, c, torch.ones(N), **kwargs)

class LagTTriDInverseAdaptiveTransition(TriDInverseAdaptiveTransition):
    def __init__(self, N, **kwargs):
        p = torch.ones(N)
        dl = _diag(N, -1.)
        du = _diag(N, 0.)
        d = torch.ones(N)
        c = torch.ones(N) * -.5

        super().__init__(N, dl, d, du, p, p, c, torch.ones(N), **kwargs)

class LegSTriDInverseAdaptiveTransition(TriDInverseAdaptiveTransition):
    def __init__(self, N, diag_scale=2, diag_add=True, **kwargs):
        # print(diag_scale, kwargs)
        if diag_scale == 2:
            p = torch.sqrt(2*torch.arange(N)+1)
        elif diag_scale == 1:
            p = torch.sqrt(torch.arange(N)+1)
        elif diag_scale == 0:
            p = torch.ones(N)
        else: raise NotImplementedError
        dl = _diag(N, -1.)
        du = _diag(N, 0.)
        d = torch.ones(N)
        if diag_add:
            c = - torch.arange(N) / (2*torch.arange(N)+1)
        else:
            c = - .5 * torch.ones(N)

        super().__init__(N, dl, d, du, p, p, c, torch.ones(N), **kwargs)
        # print(self.A)


class JacTriDInverseAdaptiveTransition(TriDInverseAdaptiveTransition):
    def __init__(self, N, halve=False, double_B=True, **kwargs):
        # print(diag_scale, kwargs)
        p = torch.sqrt(2*torch.arange(N)+2)
        dl = _diag(N, -1.)
        du = _diag(N, 0.)
        d = torch.ones(N)
        if halve:
            c = - .5 * torch.ones(N)
        else:
            c = 0.0 * torch.ones(N)

        if double_B:
            B = 2 * torch.ones(N)
        else:
            B = torch.ones(N)

        super().__init__(N, dl, d, du, p, p, c, B, **kwargs)
        # print(self.A)


class ChebITriDInverseAdaptiveTransition(TriDInverseAdaptiveTransition):
    def __init__(self, N, **kwargs):
        # p = torch.sqrt(1+2*torch.arange(N))
        p = torch.ones(N)
        dl = _diag(N, -.5) # + F.pad(torch.randn(N-1)*1e-4, (1, 1))
        du = _diag(N, .5) # + F.pad(torch.randn(N-1)*1e-4, (1, 1))
        d = torch.zeros(N) + torch.randn(N)*1e-3
        # d = torch.zeros(N)
        # d[0] += .5
        # d[-1] += .5
        dl[0] *= 2.**.5
        du[0] *= 2.**.5
        c = torch.ones(N) * 0. # + torch.randn(N)*1e-4

        super().__init__(N, dl, d, du, p, p, c, torch.ones(N), **kwargs)

class ChebIITriDInverseAdaptiveTransition(TriDInverseAdaptiveTransition):
    def __init__(self, N, **kwargs):
        p = torch.ones(N)
        du = _diag(N, .5)
        # du = 2.0 * du
        # dl = _diag(N, -.5) + F.pad(torch.randn(N-1)*2e-1, (1, 1))
        # dl = F.pad(torch.randn(N-1), (1,1)) * .5
        dl = -du
        d = torch.zeros(N) + torch.randn(N)*1e-3
        # d = torch.zeros(N)
        c = torch.ones(N) * 0. # + torch.randn(N)*1e-4

        super().__init__(N, dl, d, du, p, p, c, torch.ones(N), **kwargs)

class ToeplitzAdaptiveTransition(AdaptiveTransition):
    # NOTE stores matrix for x' = -Ax + Bu instead of x' = Ax + Bu

    def __init__(self, N, a, b, c, **kwargs):
        """Implements update for lower triangular Toeplitz transitions A.

        a: represents the diagonals of a lower triangular Toeplitz transition matrix
        b: B transition matrix
        c: scaling factors

        A = c a c^{-1}, B = c b (note that c represents \Lambda^{-1} in the HiPPO paper)
        """

        super().__init__(N, {'a': a, 'c': c, 'b': b}, **kwargs)
        e = torch.zeros(N)
        e[0] = 1.0
        self.register_buffer('e', e) # for convenience



    def _A(self): # TODO do this for all classes? how to know when to cache A or not?
        # Z = torch.diag_embed(torch.ones(self.N-1), -1).to(self.a)
        # [21-09-14 TODO] changed the krylov construction but haven't tested
        # Z = torch.diag_embed(self.ones[:-1], -1)
        # A = krylov(self.N, Z, self.a) # TODO use toeplitz.toeplitz_krylov_fast instead
        A = construct_toeplitz(self.a)
        A = A.transpose(0, 1)
        A = self.c.unsqueeze(-1) * A * self.c.reciprocal()
        return A

    # @property
    def _B(self):
        return self.c * self.b

    # TODO do we need the gbt_A() and gbt_B() methods to materialize the GBT matrices faster?

    def quadratic(self, x, y): # TODO need this? also, move to main superclass
        """
        x : (..., N)
        y : (..., N)
        returns: x^T A y (...)
        """
        return torch.sum((self.A @ y.unsqueeze(-1)).squeeze(-1) * x, dim=-1)

    def _mult(self, t, u, transpose):
        if transpose:
            x = self.c * u
            x = causal_convolution(t, x.flip(-1)).flip(-1)
            x = self.c.reciprocal() * x
        else:
            x = self.c.reciprocal() * u
            x = causal_convolution(t, x)
            x = self.c * x
        return x

    def forward_mult(self, u, delta, transpose=False):
        """Computes y = (I - delta A) u.

        self.a: (..., n)
        u: (..., n)
        delta: (...)

        x: (..., n)
        """

        t = self.e - delta.unsqueeze(-1) * self.a # represents (I - delta A)
        return self._mult(t, u, transpose)

    def inverse_mult(self, u, delta, transpose=False):
        """Computes (I + d A)^-1 u."""

        t = self.e + delta.unsqueeze(-1) * self.a
        t_ = causal_convolution_inverse(t) # represents (I + delta A)^-1
        return self._mult(t_, u, transpose)

class LagTToeplitzAdaptiveTransition(ToeplitzAdaptiveTransition):
    def __init__(self, N, **kwargs):
        a = torch.ones(N)
        a[..., 0] = .5
        b = torch.ones(N)
        c = torch.ones(N)
        super().__init__(N, a, b, c, **kwargs)

class GLagTToeplitzAdaptiveTransition(ToeplitzAdaptiveTransition):
    def __init__(self, N, alpha=0.0, beta=0.01, **kwargs):
        a = torch.ones(N)
        a[..., 0] = (1. + beta) / 2.
        # b = torch.ones(N)
        b = ss.binom(alpha + np.arange(N), np.arange(N)) * np.exp(-.5 * ss.gammaln(1-alpha)) * beta**((1-alpha)/2)
        b = torch.as_tensor(b, dtype=torch.float)
        # c = torch.ones(N)
        c = np.exp(.5 * (ss.gammaln(np.arange(N)+alpha+1) - ss.gammaln(np.arange(N)+1)))
        c = 1. / c
        c = torch.as_tensor(c, dtype=torch.float)
        super().__init__(N, a, b, c, **kwargs)
