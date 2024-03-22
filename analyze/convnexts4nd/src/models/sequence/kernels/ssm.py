"""SSM convolution kernels.

SSMKernelDPLR is the S4 kernel, implementing the 'diagonal plus low-rank' algorithm from the original S4 paper. This stores parameters A, B, C, dt, and calling it creates the SSM convolution kernel bar{K}.

SSMKernelDense is a much simpler version included for illustration purposes. It has the same output, but uses the naive SSM algorithm which is much slower. This module is meant for testing and exposition, to understand what the SSM Kernel actually does.

SSMKernelDiag is the S4D kernel, a simpler algorithm for computing the kernel for the case of diagonal state matrices A.

SSMKernel wraps these with common options and handles the initialization.
"""

from typing import Optional, Mapping, Tuple, Union
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor # For type hints
import numpy as np
from einops import rearrange, repeat

import src.models.hippo.hippo as hippo
import src.models.sequence.kernels.dplr as dplr
from src.models.functional.krylov import krylov, power
import src.utils.train

log = src.utils.train.get_logger(__name__)

# Try CUDA extension
try:
    from extensions.kernels.cauchy import cauchy_mult as cauchy_cuda
    from extensions.kernels.vandermonde import log_vandermonde_cuda
    has_cuda_extension = True
    log.info("CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) found.")
except:
    log.warning(
        "CUDA extension for structured kernels (Cauchy and Vandermonde multiplication) not found. Install by going to extensions/kernels/ and running `python setup.py install`, for improved speed and memory efficiency. Note that the kernel changed for state-spaces 4.0 and must be recompiled."
    )
    has_cuda_extension = False

try:
    import pykeops
    from src.models.functional.cauchy import cauchy_conj as cauchy_keops
    from src.models.functional.vandermonde import log_vandermonde as log_vandermonde_keops, log_vandermonde_transpose as log_vandermonde_transpose_keops

    has_pykeops = True
    log.info("Pykeops installation found.")
except ImportError:
    has_pykeops = False
    if not has_cuda_extension:
        log.warning(
            "Falling back on slow Cauchy and Vandermonde kernel. Install at least one of pykeops or the CUDA extension for better speed and memory efficiency."
        )

# Fallback versions
from src.models.functional.cauchy import cauchy_naive
from src.models.functional.vandermonde import log_vandermonde_naive
from src.models.functional.vandermonde import log_vandermonde_transpose_naive

# Base Kernel class
from src.models.sequence.kernels.kernel import Kernel

# Alias torch.einsum; can easily swap to opt_einsum if desired
contract = torch.einsum

_isnan = lambda x: torch.isnan(x).any()
_isinf = lambda x: torch.isinf(x).any()

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()

def inv_transform(param, transform='none'):
    """Initialize a (positive) parameter under a transform."""
    param = torch.clamp(param, min=1e-4)
    if transform == 'none':
        return param
    elif transform == 'exp':
        return torch.log(param) # Some of the HiPPO methods have real part 0
    elif transform == 'relu':
        return param
    elif transform == 'sigmoid':
        return torch.logit(param)
    elif transform == 'softplus':
        return torch.log(torch.exp(param)-1)
    else: raise NotImplementedError

def param_transform(param, transform='none'):
    """Get a (positive) parameter under a transform."""
    if transform == 'none':
        p = param
    elif transform == 'exp':
        p = torch.exp(param)
    elif transform == 'relu':
        # JAX version seems to NaN if you allow 0's, although this code was fine without it
        p = F.relu(param)+1e-4
    elif transform == 'sigmoid':
        p = F.sigmoid(param)
    elif transform == 'softplus':
        p = F.softplus(param)
    else: raise NotImplementedError
    return p


class SSMKernel(Kernel):
    """Parent class for different SSM parameterizations.

    This class is abstract and only defines some initializations and flags that are common to all SSM variants.
    It is instantiated by subclasses SSMKernel{Dense,Real,Diag,DPLR}.

    Options:
    d_state (N): State size (dimensionality of parameters A, B, C). Generally shouldn't need to be adjusted and doens't affect speed much for most kernels (e.g. S4, S4D).
    deterministic: Use a deterministic initialization for dt, A, B, C.
        Useful for debugging as well as constructing a simple exponential decay kernel (e.g. used in S4ND image->video inflation).

    dt_min, dt_max: min and max values for the step size dt
    dt_tie: Keep dt tied across the N dimensions of the state. Although this theoretically makes more sense, models such as S5 and Mega have found slightly improvements by setting it to False.
    dt_transform: Transform function for parameterization of dt (default 'softplus', used to be 'exp')

    rank: Rank of low-rank correction for DPLR mode. Needs to be increased for init "legt".
    n_ssm: Number of independent trainable (A, B) SSMs, e.g.
        `n_ssm=1` means all A/B parameters are tied across the H different instantiations of C.
        `n_ssm=None` means all H SSMs are completely independent.
        Generally, changing this option can save parameters but doesn't affect performance or speed much.
        This parameter must divide H.
    init: Options for initialization of (A, B). For DPLR mode, recommendations are "legs", "fout", "hippo" (combination of both). For Diag mode, recommendations are "diag-inv", "diag-lin", "diag-legs", and "diag" (combination of diag-inv and diag-lin).
    init_args: Extra arguments passed into initialization function (see dplr.py for options).
    """

    def init_dt(self):
        # Generate dt
        if self.deterministic:  # Meant for debugging
            assert self.dt_tie, "Deterministic dt initialization is tied"
            assert self.dt_transform == 'exp', "Deterministic dt transform should be 'exp' for simplicity"
            inv_dt = torch.exp(torch.linspace(math.log(self.dt_min), math.log(self.dt_max), self.H)).unsqueeze(-1) # (H 1)
        else:
            shape = (self.H, 1) if self.dt_tie else (self.H, self.N//2)
            # Initialize log dt
            inv_dt = torch.rand(*shape, dtype=self.dtype) * (
                math.log(self.dt_max) - math.log(self.dt_min)
            ) + math.log(self.dt_min)
            if self.dt_transform != 'exp':
                inv_dt = inv_transform(torch.exp(inv_dt), self.dt_transform)

        return inv_dt

    def init_ssm_real(self):
        """Returns (dense, real) (A, B, C) parameters for init options."""
        # Generate A, B
        A, B = hippo.transition(self.init, self.N)
        A = torch.as_tensor(A, dtype=self.dtype)
        B = torch.as_tensor(B, dtype=self.dtype)[:, 0]
        B = repeat(B, 'n -> v n', v=self.n_ssm).clone().contiguous()
        A = repeat(A, 'n m -> v n m', v=self.n_ssm).clone().contiguous()

        # Generate C
        if self.deterministic:
            C = torch.zeros(self.channels, self.H, self.N, dtype=self.dtype)
            C[..., :1] = 1.0
        else:
            C = torch.randn(self.channels, self.H, self.N, dtype=self.dtype)

        return A, B, C

    def init_ssm_dplr(self):
        """Returns DPLR (A, P, B, C) parameters for init options."""
        A, P, B, V = dplr.combination(self.init, self.N, self.rank, self.n_ssm, **self.init_args)

        # Broadcast C to have H channels
        if self.deterministic:
            C = torch.zeros(self.channels, self.n_ssm, self.N, dtype=self.cdtype)
            C[:, :, :1] = 1.
            C = contract('hmn, chn -> chm', V.conj().transpose(-1, -2), C) # V^* C
            C = repeat(C, 'c t n -> c (v t) n', v=self.H // C.size(-2)).clone().contiguous()
        else:
            C = torch.randn(self.channels, self.H, self.N//2, dtype=self.cdtype)

        # Broadcast other parameters to have n_ssm copies
        assert self.n_ssm % B.size(-2) == 0 \
                and self.n_ssm % P.size(-2) == 0 \
                and self.n_ssm % A.size(-2) == 0

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        P = repeat(P, 'r t n -> r (v t) n', v=self.n_ssm // P.size(-2)).clone().contiguous()
        A = repeat(A, 't n -> (v t) n', v=self.n_ssm // A.size(-2)).clone().contiguous()

        # Because these complex parameterizations assume conjugate symmetry,
        # halve the value of self.N for convenience
        self.N //= 2

        return A, P, B, C

    def __init__(
        self,
        # General Kernel arguments for parent class
        d_model: int = 0,
        channels: int = 1,
        l_max: Optional[int] = None,
        lr: Union[float, Optional[Mapping]] = None,
        wd: Union[float, Optional[Mapping]] = 0.0,
        verbose: bool = True,
        # SSM arguments
        d_state: int = 64,
        deterministic: bool = False,
        # dt options
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_tie: bool = True,
        dt_transform: str = 'exp',
        # (A, B, C) options
        rank: int = 1,
        n_ssm: Optional[int] = None,
        measure: Optional[str] = None,
        init: Optional[str] = "legs",
        # Extra hyperparameters for initialization
        **init_args,
    ):
        super().__init__(d_model=d_model, channels=channels, l_max=l_max, lr=lr, wd=wd, verbose=verbose)
        self.N = d_state
        self.dtype, self.cdtype = torch.float, torch.cfloat
        self.deterministic = deterministic
        # dt options
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_tie = dt_tie
        self.dt_transform = dt_transform
        # SSM options (A, B, C)
        self.rank = rank
        self.n_ssm = n_ssm if n_ssm is not None else self.H
        if measure is not None:
            log.warning("Warning: 'measure' option changed to 'init' and will be removed in a future version.")
            assert init is None, "'measure' and 'init' cannot both be passed into SSMKernel"
            init, measure = measure, init
        self.init = init
        self.init_args = init_args

    @torch.no_grad()
    def forward_state(self, u, state):
        """Forward the state through a sequence, i.e. computes the state after passing chunk through SSM

        This is a generic version of this functionality that works for SSMs.
        It is currently used by SSMKernelDense and SSMKernelDPLR.
        This is a suboptimal implementation; it is recommended to use SSMKernelDiag
        if this functionality is desired.

        state: (B, H, N)
        u: (B, H, L)

        Returns: (B, H, N)
        """

        # Construct dA, dB matrices
        dA, dB = self._setup_state() # (H N N) (H N)

        conj = state.size(-1) != dA.size(-1)
        if conj: state = _conj(state)

        v = contract('h n, b h l -> b h n l', dB, u.flip(-1))
        AL, v = power(u.size(-1), dA, v)
        next_state = contract("h m n, b h n -> b h m", AL, state)
        next_state = next_state + v

        if conj: next_state = next_state[..., : next_state.size(-1) // 2]
        return next_state

    def _setup_state(self):
        """Register dA and dB to module."""
        raise NotImplementedError

    @property
    def d_state(self):
        """d_state and state_to_tensor are used by specific decoders.

        These were used in earlier versions and should not be needed in general.
        """
        return self.H * self.N

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)

class SSMKernelDense(SSMKernel):
    """Slow version of SSMKernel function for illustration and benchmarking.

    Uses dense A parameterization and computes kernel in naive way.
    - Discretize A^(dt), B^(dt) using bilinear transform
    - Compute length-L kernel K_L(A^(dt), B^(dt), C)
    """

    @staticmethod
    def bilinear(dt, A, B=None):
        """
        dt: (H 1) timescales (or H N)
        A: (H N N)
        B: (H N)
        """
        N = A.shape[-1]
        I = torch.eye(N).to(A)
        A_backwards = I - dt[:, None] / 2 * A # Doesn't quite make sense if dt has shape (H N)
        A_forwards = I + dt[:, None] / 2 * A

        if B is None:
            dB = None
        else:
            dB = dt * torch.linalg.solve(
                A_backwards, B.unsqueeze(-1)
            ).squeeze(-1) # (... N)

        dA = torch.linalg.solve(A_backwards, A_forwards)  # (... N N)
        return dA, dB

    def __init__(self, comp=False, **kwargs):
        """
        comp: Use Companion matrix parameterization (SpaceTime).
        """
        super().__init__(**kwargs)
        self.comp = comp

        # Initialize dt, A, B, C
        inv_dt = self.init_dt()
        A, P, B, C = self.init_ssm_dplr()

        # Materialize dense A, B, C
        if self.comp:
            # Special case for companion matrix parameterization
            A = torch.zeros_like(_conj(A))
        else:
            A = torch.diag_embed(_conj(A)) \
                - contract('r s p, r s q -> s p q', _conj(P), _conj(P).conj())
        self.N *= 2  # Double N again since no conjugate symmetry
        B, C = _conj(B), _conj(C)


        self.register_params(A, B, C, inv_dt)

    def register_params(self, A, B, C, inv_dt):
        assert self.N == A.size(-1)
        assert self.H == inv_dt.size(0)
        assert self.n_ssm == A.size(0) == B.size(0)
        self.repeat = self.H // A.size(0)

        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)

        # Register parameters
        self.register("inv_dt", inv_dt, self.lr_dict['dt'], self.wd_dict['dt'])
        self.register("A", _c2r(A), self.lr_dict['A'], self.wd_dict['A'])
        self.register("B", _c2r(B), self.lr_dict['A'], self.wd_dict['B'])
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        # Cache if nothing is trained
        is_trainable = lambda lr: lr is None or lr > 0.0
        self.trainable = is_trainable(self.lr_dict['dt']) \
                or is_trainable(self.lr_dict['A']) \
                or is_trainable(self.lr_dict['B'])
        self.K = None # Compute in forward pass since that ensures correct device

    def forward(self, state=None, rate=1.0, L=None):
        if L is None: L = self.L
        # This class shouldn't support the more advanced sampling and variable length functionalities, since it's just for testing
        # But the code from NPLR could be pasted here if desired
        # assert rate == 1.0 and L is not None

        if self.trainable or self.K is None:
            dA, dB = self._setup_state()
            self.dA, self.dB = dA, dB
            # Need to calculate dA, dB

        if self.trainable:
            k = krylov(L, self.dA, self.dB, _r2c(self.C))  # (H L)
        else:
            if self.K is None:
                self.K = krylov(L, self.dA, self.dB) # (H N L)
            k = contract('hnl,chn->chl', self.K[..., :L], _r2c(self.C))
        k = k.float()

        if state is not None:
            state = state.to(self.dA)
            # Compute A @ s
            state = contract("h n m, b h m -> b h n", self.dA, state)
            k_state = krylov(L, self.dA, state.unsqueeze(-3), _r2c(self.C))
            k_state = k_state.float()
        else:
            k_state = None
        return k, k_state

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def _setup_state(self):
        A, B = _r2c(self.A), _r2c(self.B)
        A = repeat(A, 't n m -> (v t) n m', v=self.repeat)
        B = repeat(B, 't n -> (v t) n', v=self.repeat)
        if self.comp:
            dA = A.new_zeros((self.H, self.N, self.N))
            dA[:, 1:, :-1] = torch.eye(self.N-1, dtype=A.dtype, device=A.device)
            # A = A/torch.linalg.norm(A,ord=1,dim=-1,keepdims=True)
            dA[:, :, -1] = A
            dB = _r2c(self.B).expand((self.H, self.N))
            dA = dA.real + 0j
            dB = dB.real + 0j
        else:
            dt = param_transform(self.inv_dt, self.dt_transform)
            dA, dB = SSMKernelDense.bilinear(dt, A, B)
        return dA, dB

    def _setup_step(self):
        self.dA, self.dB = self._setup_state()
        self.dC = _r2c(self.C)

    def step(self, u, state):
        next_state = contract("h m n, b h n -> b h m", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return y.real, next_state

class SSMKernelReal(SSMKernelDense):
    """Dense and real version of SSMKernel (e.g. using original real-valued HiPPO matrices) for testing."""
    def __init__(self, **kwargs):
        super().__init__(comp=False, **kwargs)

        inv_dt = self.init_dt()
        A, B, C = self.init_ssm_real()

        # SSMKernelDense is designed to work with complex
        A, B, C = A.to(torch.cfloat), B.to(torch.cfloat), C.to(torch.cfloat)
        self.register_params(A, B, C, inv_dt)


class SSMKernelDiag(SSMKernel):
    """SSM kernel using diagonal state matrix (S4D model).

    Options:
    disc: ['zoh' | 'bilinear' | 'dss'] Discretization options.
    dt_fast:  (experimental) Parameterize inv_dt under sinh function.
        (Ohno et al. "Fast Saturating Gate for Learning Long Time Scales with RNNs")
    real_transform, imag_transform: ['none' | 'exp' | 'relu' | 'sigmoid' | 'softplus']
        Parameterize the real/imag parts of the diagonal of A under this function.
    bandlimit: Mask high frequencies of the kernel (indices corresponding to
        diagonal elements with large imaginary part). Introduced in S4ND paper.
    backend: ['cuda' | 'keops' | 'naive'] Options for Vandermonde/Cauchy kernel (in order of efficiency).
    is_real : Real-valued SSM; can be interpreted as EMA.
    """

    def __init__(
        self,
        disc: str = 'zoh',  # Change to 'bilinear' to match S4, but should make little difference either way
        dt_fast: bool = False,
        real_transform: str = 'exp',
        imag_transform: str = 'none',
        bandlimit: Optional[float] = None,
        backend: str = 'cuda',
        is_real: bool = False,
        **kwargs,
    ):
        # Special case: for real-valued, d_state semantics change
        if is_real and 'd_state' in kwargs:
            kwargs['d_state'] = kwargs['d_state'] * 2
        super().__init__(**kwargs)
        self.disc = disc
        self.dt_fast = dt_fast
        self.real_transform = real_transform
        self.imag_transform = imag_transform
        self.bandlimit = bandlimit
        self.backend = backend
        self.is_real = is_real

        # Initialize dt, A, B, C
        inv_dt = self.init_dt()
        A, P, B, C = self.init_ssm_dplr()
        # Note that in the Diag case, P will be ignored
        # The DPLR case subclasses this and uses P
        self.register_params(A, B, C, inv_dt, P)

    def register_params(self, A, B, C, inv_dt, P):
        """Process the initialization into form of trainable parameters.

        A: (S, N) diagonal matrix
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        assert self.backend in ['cuda', 'keops', 'naive']

        if self.dt_fast: inv_dt = torch.asinh(inv_dt)

        # Rank of low-rank correction
        assert self.H == inv_dt.size(0)
        assert self.N == A.size(-1) == B.size(-1) == C.size(-1)
        assert self.n_ssm == A.size(-2) == B.size(-2) # Number of independent SSMs trained
        self.repeat = self.H // A.size(0)

        # Check that diagonal part has negative real and imag part
        # (allow some tolerance for numerical precision on real part
        # since it may be constructed by a diagonalization)
        assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (C, H, N)  # TODO originally this was only in DPLR, check safe for Diag
        B = B.unsqueeze(0) # (1, H, N)
        assert self.channels == C.shape[0]

        # Register dt
        self.register("inv_dt", inv_dt, self.lr_dict['dt'], self.wd_dict['dt'])
        # Register ABC
        if self.is_real:
            self.register("C", C.real, self.lr_dict['C'], None)
            self.register("B", B.real, self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
        else:
            self.register("C", _c2r(_resolve_conj(C)), self.lr_dict['C'], None)
            self.register("B", _c2r(B), self.lr_dict['B'], self.wd_dict['B'])
            self.register("A_real", inv_transform(-A.real, self.real_transform), self.lr_dict['A'], self.wd_dict['A'])
            self.register("A_imag", inv_transform(-A.imag, self.imag_transform), self.lr_dict['A'], self.wd_dict['A'])

    def _get_params(self, rate=1.0):
        """Process the internal parameters."""

        # (S N) where S=n_ssm
        if self.is_real:
            A = -param_transform(self.A_real, self.real_transform)
            B = self.B # (1 S N)
            C = self.C # (C H N)
        else:
            A = -param_transform(self.A_real, self.real_transform) - 1j * param_transform(self.A_imag, self.imag_transform)
            B = _r2c(self.B) # (1 S N)
            C = _r2c(self.C) # (C H N)

        if self.dt_fast: inv_dt = torch.sinh(self.inv_dt)
        else: inv_dt = self.inv_dt
        dt = param_transform(inv_dt, self.dt_transform) * rate # (H N)

        if self.bandlimit is not None:
            freqs = dt / rate * A.imag.abs() / (2*math.pi) # (H N)
            mask = torch.where(freqs < self.bandlimit * .5, 1, 0)
            C = C * mask

        # Incorporate dt into A and B
        A = repeat(A, 't n -> (v t) n', v=self.repeat)  # (H N)
        B = repeat(B, 'b t n -> b (v t) n', v=self.repeat)  # (1 H N)

        # TODO: The downstream algorithm should only need to access dt*A
        # However the current DPLR kernel still uses dt and A separately
        # Once that is fixed, this should return dtA instead of dt and A
        dtA = dt * A  # (H N)

        return dt, A, B, C

    def forward(self, L, state=None, rate=1.0):
        """See Kernel.forward() for argument documentation."""

        dt, A, B, C = self._get_params(rate)
        dtA = dt * A

        # Augment B with state
        if state is not None:
            s = state / dt
            if self.disc == 'bilinear':
                s = s * (1. + dtA/2)
            elif self.disc == 'zoh':
                s = s * dtA * dtA.exp() / (dtA.exp() - 1.)
            B = torch.cat([s, B], dim=-3) # (1+B H N)


        # Combine B and C
        C = (B[:, None, :, :] * C).view(-1, self.H, self.N)

        # Dispatch which Vandermonde kernel to use
        if has_cuda_extension and C.dtype == torch.cfloat and C.device.type == 'cuda' and self.backend == 'cuda':
            log_vandermonde = log_vandermonde_cuda
        elif has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde = log_vandermonde_keops
        else:
            log_vandermonde = log_vandermonde_naive

        # Main kernel
        if self.disc == 'zoh':
            # Power up
            C = C * (torch.exp(dtA)-1.) / A
            K = log_vandermonde(C, dtA, L) # (H L)
        elif self.disc == 'bilinear':
            C = C * (1. - dtA/2).reciprocal() * dt # or * dtA / A
            dA = (1. + dtA/2) / (1. - dtA/2)
            K = log_vandermonde(C, dA.log(), L)
        elif self.disc == 'dss':
            # Implementation from DSS meant for case when real eigenvalues can be positive
            P = dtA.unsqueeze(-1) * torch.arange(L, device=C.device) # [H N L]
            A_gt_0 = A.real > 0                                      # [N]
            if A_gt_0.any():
                with torch.no_grad():
                    P_max = dtA * (A_gt_0 * (L-1))                   # [H N]
                P = P - P_max.unsqueeze(-1)                          # [H N L]
            S = P.exp()                                              # [H N L]

            dtA_neg = dtA * (1 - 2*A_gt_0)                           # [H N]
            num = dtA_neg.exp() - 1                                  # [H N]
            den = (dtA_neg * L).exp() - 1                            # [H N]

            # Inline reciprocal function for DSS logic
            x = den * A
            x_conj = _resolve_conj(x)
            r = x_conj / (x*x_conj + 1e-7)

            C = C * num * r             # [C H N]
            K = contract('chn,hnl->chl', C, S).float()
        else: raise ValueError(f"Discretization {self.disc} not supported")

        K = K.view(-1, self.channels, self.H, L) # (1+B C H L)

        if state is not None:
            K_state = K[:-1, :, :, :] # (B C H L)
        else:
            K_state = None
        K = K[-1, :, :, :] # (C H L)

        return K, K_state

    def _setup_step(self):
        """Set up dA, dB, dC discretized parameters for stepping."""

        dt, A, B, C, = self._get_params()
        # Incorporate dt into A
        dtA = dt * A  # (H N)

        if self.disc == 'zoh':
            self.dA = torch.exp(dtA) # (H N)
            self.dB = B * (torch.exp(dtA)-1.) / A # (C H N)
        elif self.disc == 'bilinear':
            self.dA = (1. + dtA/2) / (1. - dtA/2)
            self.dB = B * (1. - dtA/2).reciprocal() * dt # or * dtA / A
        self.dB = rearrange(self.dB, '1 h n -> h n')
        self.dC = C

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state

    def forward_state(self, u, state):
        """Pass the state forward through an entire sequence."""
        self._setup_step()
        AL = self.dA ** u.size(-1)
        u = u.flip(-1).to(self.dA).contiguous() # (B H L)
        # Dispatch which Vandermonde kernel to use
        if has_pykeops and self.backend in ['cuda', 'keops']:
            log_vandermonde_transpose = log_vandermonde_transpose_keops
        else:
            log_vandermonde_transpose = log_vandermonde_transpose_naive
        v = log_vandermonde_transpose(u, self.dB, self.dA.log(), u.size(-1))
        next_state = AL * state + v
        return next_state

class SSMKernelDPLR(SSMKernelDiag):
    """SSM kernel for diagonal + low rank (DPLR) state matrices, corresponding to the original S4 model."""

    @torch.no_grad()
    def _setup_C(self, L):
        """Construct C~ from C.

        Two modes are supported: go directly to length L if self.l_kernel is 1, or length is doubled
        """

        if self.l_kernel.item() == 0:
            if self.verbose: log.info(f"S4: Initializing kernel to length {L}")
            double_length = False
        elif L > self.l_kernel.item(): # 2*int(self.l_kernel) == L:
            if self.verbose: log.info(f"S4: Doubling length from L = {self.l_kernel.item()} to {2*self.l_kernel.item()}")
            double_length = True
            L = self.l_kernel.item() # Convenience for the math below
        else: return

        C = _r2c(self.C)
        dA, _ = self._setup_state()
        dA_L = power(L, dA)
        # Multiply C by I - dA_L
        C_ = _conj(C)
        prod = contract("h m n, c h n -> c h m", dA_L.transpose(-1, -2), C_)
        if double_length: prod = -prod # Multiply by I + dA_L instead
        C_ = C_ - prod
        C_ = C_[..., :self.N] # Take conjugate pairs again
        self.C.copy_(_c2r(C_))

        self.l_kernel = 2*self.l_kernel if double_length else self.l_kernel+L # Preserve type/device

    def _omega(self, L, dtype, device, cache=True):
        """Calculate (and cache) FFT nodes.

        This also caches a version of the nodes "unprocessed" with the bilinear transform.
        This method should be called everytime the internal length self.l_kernel changes.
        """

        # Use cached if available
        if cache and hasattr(self, 'omega') and self.omega.size(-1) == L//2+1:
            return self.omega, self.z

        omega = torch.tensor(
            np.exp(-2j * np.pi / (L)), dtype=dtype, device=device
        )  # \omega_{2L}
        omega = omega ** torch.arange(0, L // 2 + 1, device=device)
        z = 2 * (1 - omega) / (1 + omega)

        # Cache if necessary
        if cache:
            self.omega = omega
            self.z = z
        return omega, z


    def register_params(self, A, B, C, inv_dt, P):
        """Process the initialization into form of trainable parameters.

        The SSM state matrix is represented by diag_embed(A) - PP^*
        Note that the A notation here is slightly overloaded:
        normally A refers to the full SSM state matrix (DPLR in this case)
        but here we're using it to refer to the diagonal part of the matrix.
        This is to make variable names compatible with the SSMKernelDiag class (DSS/S4D)
        and is a much simpler variable name (e.g. as opposed to Lambda).

        A: (S, N) diagonal part
        P: (R, S, N) low-rank part
        B: (S, N)
        C: (C, H, N)
        dt: (H) timescale per feature

        Dimensions:
        N (or d_state): state size
        H (or d_model): total SSM copies
        S (or n_ssm): number of trainable copies of (A, B, dt); must divide H
        R (or rank): rank of low-rank part
        C (or channels): system is 1-dim to C-dim

        The forward pass of this Module returns a tensor of shape (C, H, L)

        Note: tensor shape N here denotes half the true state size, because of conjugate symmetry
        """

        # Print out kernel lengths; it can be tricky to make sure the length logic is correct
        if self.verbose:
            log.info(f"Constructing S4 (H, N, L) = ({self.H}, {self.N}, {self.l_max})")

        # Register the basic params for diagonal SSM (A, B, C, dt)
        super().register_params(A, B, C, inv_dt, P)

        # Check shapes
        assert self.rank == P.shape[-3]
        assert self.N == P.size(-1)
        assert self.n_ssm == P.size(-2)

        self.register('P', _c2r(P), self.lr_dict['A'], self.wd_dict['A'])

        # Track the current kernel length this is "attuned" to
        self.register_buffer('l_kernel', torch.tensor(0))

    def _get_params(self, rate=1.0):
        dt, A, B, C = super()._get_params(rate=rate)
        P = _r2c(self.P)  # (R S N)
        P = repeat(P, 'r t n -> r (v t) n', v=self.repeat)  # (R H N)
        Q = P.conj()

        return dt, A, B, C, P, Q

    def forward(self, state=None, rate=1.0, L=None):
        """See Kernel.forward() for argument documentation."""

        # Initialize C~ if necessary (done in forward pass so it's on the correct device)
        if self.l_kernel.item() == 0 and self.l_max is not None and self.l_max > 0:
            self._setup_C(self.l_max)

        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.l_kernel, while we are asked to provide a kernel of length L at (relative) frequency rate
        if L is None:
            L = round(self.l_kernel.item() / rate)

        # Increase the internal length if needed
        continuous_L = round(rate*L)
        while continuous_L > self.l_kernel.item():
            self._setup_C(continuous_L)
        discrete_L = round(self.l_kernel.item()/rate)

        dt, A, B, C, P, Q = self._get_params(rate)

        # Get FFT nodes of right length
        omega, z = self._omega(discrete_L, dtype=A.dtype, device=A.device, cache=(rate==1.0))

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state # (B H N)
            sA = (
                s * _conj(A) # (B H N)
                - contract('bhm, rhm, rhn -> bhn', s, _conj(Q), _conj(P))
            )
            s = s / dt + sA / 2
            s = s[..., :self.N]

            B = torch.cat([s, B], dim=-3)  # (B+1, H, N)

        # Incorporate dt into A
        A = A * dt  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3) # (B+1+R, H, N)
        C = torch.cat([C, Q], dim=-3) # (C+R, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (B+1+R, C+R, H, N)
        v = v * dt  # Incorporate dt into B

        # Dispatch which Cauchy kernel to use
        if has_cuda_extension and z.dtype == torch.cfloat and z.device.type == 'cuda' and self.backend == 'cuda':
            cauchy_mult = cauchy_cuda
        elif has_pykeops and self.backend in ['cuda', 'keops']:
            cauchy_mult = cauchy_keops
        else:
            cauchy_mult = cauchy_naive
        # Calculate resolvent at omega
        r = cauchy_mult(v, z, A)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (1 + r[-1:, -1:, :, :])
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[:1, 1:, :, :] * r11[1:, :1, :, :]
            s = (
                r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
                + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
                - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
                - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
            )
            s = s / det
            k_f = r00 - s
        else:
            r00 = r[:-self.rank, :-self.rank, :, :]
            r01 = r[:-self.rank, -self.rank:, :, :]
            r10 = r[-self.rank:, :-self.rank, :, :]
            r11 = r[-self.rank:, -self.rank:, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f, n=discrete_L)  # (B+1, C, H, L)

        # # Truncate to target length
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (B, C, H, L)
        else:
            k_state = None
        k_B = k[-1, :, :, :] # (C H L)

        return k_B, k_state

    @torch.no_grad()
    def double_length(self):
        self._setup_C(2*self.l_kernel)

    @torch.no_grad()
    def _check(self):
        """Check if A, B, C parameters and vanilla SSMKernel construction can be recovered"""

        # assert self.l_kernel > 0, "Set up module first"

        K = self.forward(L=self.l_max)[0]

        self._setup_step()
        K_ = krylov(self.l_max, self.dA, self.dB, self.dC)

        diff = K - K_
        print("checking DPLR Kernel construction", torch.sum(diff ** 2))

    @torch.no_grad()
    def _setup_linear(self):
        """Preprocessing that allows fast linear-time (in state dimension) stepping."""
        dt, A, B, C, P, Q = self._get_params()

        # Prepare Linear stepping
        D = (2.0 / dt - A).reciprocal()  # (H, N)
        R = (torch.eye(self.rank, dtype=A.dtype, device=A.device) + 2*contract('r h n, h n, s h n -> h r s', Q, D, P).real) # (H R R)
        Q_D = rearrange(Q*D, 'r h n -> h r n')
        try:
            R = torch.linalg.solve(R, Q_D) # (H R N)
        except:
            R = torch.tensor(np.linalg.solve(R.to(Q_D).contiguous().detach().cpu(), Q_D.contiguous().detach().cpu())).to(Q_D)
        R = rearrange(R, 'h r n -> r h n')

        self.step_params = {
            "D": D, # (H N)
            "R": R, # (R H N)
            "P": P, # (R H N)
            "Q": Q, # (R H N)
            "B": B, # (1 H N)
            "E": 2.0 / dt + A, # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations.
        Perhaps a fused CUDA kernel implementation would be much faster.

        u: (H) Input
        state: (H, N/2) State with conjugate pairs. Optionally, the state can have last dimension N.

        Returns: same shape as state
        """
        C = _r2c(self.C) # View used for dtype/device

        if u is None: # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None: # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if state.size(-1) == self.N: # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', _conj(p), _conj(x), _conj(y))[..., :self.N] # inner outer product
        else:
            assert state.size(-1) == 2*self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            contract_fn = lambda p, x, y: contract('r h n, r h m, ... h m -> ... h n', p, x, y) # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (R H N)
        P = step_params["P"]  # (R H N)
        Q = step_params["Q"]  # (R H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state) # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation."""

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C) # Just returns a view that we use for finding dtype/device

        state = torch.eye(2*self.N, dtype=C.dtype, device=C.device).unsqueeze(-2) # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        dB = rearrange(dB, '1 h n -> h n') # (H N)
        return dA, dB

    def _step_state(self, u, state):
        """Must be called after self.default_state() is used to construct an initial state!"""
        next_state = (torch.einsum(self.state_contraction, self.dA, state)
                     + torch.einsum(self.input_contraction, self.dB, u))
        return next_state

    def _setup_step(self, mode='dense'):
        """Set up dA, dB, dC discretized parameters for stepping."""
        self.dA, self.dB = self._setup_state()

        # Calculate original C
        C = _conj(_r2c(self.C)) # (H C N)
        if self.l_kernel.item() == 0:
            dC = C
        else:
            # self.C represents C_tilde
            dA_L = power(self.l_kernel.item(), self.dA)
            I = torch.eye(self.dA.size(-1)).to(dA_L)

            dC = torch.linalg.solve(
                I - dA_L.transpose(-1, -2),
                C.unsqueeze(-1),
            ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode
        if mode == 'linear':
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2*self.dC[:, :, :self.N]
        elif mode == 'diagonal':
            # Eigendecomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigendedecomposition is correct
            if self.verbose:
                print("Diagonalization error:", torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA))

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract('h n m, h m -> h n', V_inv, self.dB)
            self.dC = contract('h n m, c h n -> c h m', V, self.dC)

        elif mode == 'dense':
            pass
        else: raise NotImplementedError("DPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}")

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        step_mode = getattr(self, "_step_mode", "dense")  # Used in default_state, which is called without _setup_step() in forward_state()
        if step_mode != 'linear':
            N *= 2

            if step_mode == 'diagonal':
                self.state_contraction = "h n, ... h n -> ... h n"
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = "h m n, ... h n -> ... h m"

            self.input_contraction = "h n, ... h -> ... h n"

        self.output_contraction = "c h n, ... h n -> ... c h"

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """Must have called self._setup_step() and created state with self.default_state() before calling this."""

        if self._step_mode == 'linear':
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = torch.einsum(self.output_contraction, self.dC, new_state)
        return y.real, new_state

    def forward_state(self, *args, **kwargs):
        # Dispatch directly to generic state forwarding
        # instead of using the Diag version

        # TODO design pattern is ugly. Can be fixed with an intermediate
        # subclass above Diag/DPLR that has the shared logic (parameter construction)
        # but not the state/step logic.
        # Fine to keep like this for now since we want Diag to be the standard
        # instead of having too many layers of subclassing.

        return SSMKernel.forward_state(self, *args, **kwargs)
