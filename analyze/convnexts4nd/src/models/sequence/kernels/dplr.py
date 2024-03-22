"""Initializations of structured state space (S4) models with diagonal plus low rank (DPLR) parameterization."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import src.models.hippo.hippo as hippo

import src.utils.train
log = src.utils.train.get_logger(__name__)

def dplr(
    init='hippo',
    N=64, rank=1, H=1,
    dtype=torch.float,
    real_random=False,
    real_scale=1.0,
    imag_random=False,
    imag_scale=1.0,
    B_random=False,
    B_init='constant',
    B_scale=1.0,
    P_scale=1.0,
    normalize=False,
):
    """Directly construct a DPLR matrix.

    Args:
    - init: (str) ['rand', 'lin', inv', 'real', 'hippo'] Choices for initialization of A.
          Most of these affect the imaginary part of A, except for 'real'.
    - real_random: (bool) Initialize A.real in -U[0, 1]. Otherwise, initialize to -1/2.
    - real_scale: (float) Scaling factor of real part of A.
    - imag_random: (bool) Initialize A.imag randomly.
    - imag_scale: (bool) Scaling factor of imaginary part of A.
    - B_init: (str) ['constant' | 'random' | 'alternating' | 'unit-cw' | 'unit-ccw' | 'hippo']
          Choices for initialization of B.
    - B_scale: (float) Scaling factor for B
    - P_scale: (float) Scaling factor for P
    - normalize: (bool) Apply an automatic normalization factor on B
    """
    assert dtype == torch.float or dtype == torch.double
    dtype = torch.cfloat if dtype == torch.float else torch.cdouble

    pi = torch.tensor(math.pi)

    # Construct real part of diagonal A (must be non-negative)
    if real_random:
        real_part = torch.rand(H, N//2)
    else:
        real_part = .5 * torch.ones(H, N//2)
    real_part = real_scale * real_part

    # Construct imaginary part of diagonal A (must be non-negative)
    if imag_random:
        imag_part = N//2 * torch.rand(H, N//2)
    else:
        imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)

    if init in ['random', 'rand']:
        imag_part = torch.exp(torch.randn(H, N//2))
    elif init == 'real':
        imag_part = 0 * imag_part
        if real_random:
            real_part = torch.rand(H, N//2) * N//2
        else:
            # This is the S4D-Real method described in the S4D paper
            # The A matrix is diag(-1, -2, ..., -N), which are the eigenvalues of the HiPPO matrix
            real_part = 1 + repeat(torch.arange(N//2), 'n -> h n', h=H)
    elif init in ['linear', 'lin']:
        imag_part = pi * imag_part
    elif init in ['inverse', 'inv']: # Based on asymptotics of the default HiPPO matrix
        imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    elif init in ['inverse2', 'inv2']:
        imag_part = 1/pi * N * (N/(1+imag_part)-1)
    elif init in ['quadratic', 'quad']:
        imag_part = 1/pi * (1+2*imag_part)**2
    elif init in ['legs', 'hippo']:
        A, _, _, _ = hippo.nplr('legs', N)
        imag_part = -A.imag  # Positive
    else: raise NotImplementedError
    imag_part = imag_scale * imag_part

    # Construct diagonal A
    A = -real_part - 1j * imag_part  # Force negative real and imag
    assert torch.all(A.real < 1e-4) and torch.all(A.imag <= 0.0)  # Allow some tolerance for numerical precision on real part

    # Initialize B
    if B_random:
        log.warning("'B_random' is deprecated in favor of B_init='random' and will be deprecated in a future version.")
    if init in ['legs', 'hippo']:
        log.info(f'Initializing with S4D-LegS and ignoring argument {B_init=}')
        B_init = 'legs'

    if B_init in ['legs', 'hippo']:
        # Special initialization using the HiPPO B matrix
        # Note that theory (from S4D paper) says that B should be halved
        # to match DPLR but we drop this 0.5 factor for simplicity
        _, P, B, _ = hippo.nplr('legs', N, B_clip=2.0)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'constant':
        B = torch.ones(H, N//2, dtype=dtype)
    elif B_init == 'random':
        B = torch.randn(H, N//2, dtype=dtype)
    elif B_init == 'alternating':  # Seems to track 'constant' exactly for some reason
        B = torch.ones(H, N//4, 2, dtype=dtype)
        B[:, :, 1] *= -1
        B = B.view(H, N//2)
    elif B_init == 'unit-cw':
        z = torch.tensor(torch.exp(-2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    elif B_init == 'unit-ccw':
        z = torch.tensor(torch.exp(2j * pi / N), dtype=dtype)
        B = z ** torch.arange(0, N // 2)
        B = repeat(B, 'n -> h n', h=H).clone().contiguous()
    else: raise NotImplementedError
    B *= B_scale

    # Experimental feature that appeared in earlier versions of HTTYH (not extensively tested)
    # Seems more principled for normalization theoretically, but seemed to hurt on PathX
    if normalize:
        norm = -B/A # (H, N) # Result if you integrate the kernel with constant 1 function
        zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
        B = B / zeta**.5

    # Initialize P
    if B_init in ['legs', 'hippo']:
        # P constructed earlier
        P = repeat(P, 'r n -> r h n', h=H).clone().contiguous()
    else:
        P = torch.randn(rank, H, N//2, dtype=dtype)
        P = P * P_scale

    # Initialize V (only used in testing)
    V = torch.eye(N, dtype=dtype)[:, :N//2]
    V = repeat(V, 'n m -> h n m', h=H)

    return A, P, B, V

def ssm(init, N, R, H, **ssm_args):
    """Dispatcher to create single SSM initialization

    N: state size
    R: rank (for DPLR parameterization)
    H: number of independent SSM copies
    """

    if init.startswith("diag") or init.startswith("dplr"):
        if init.startswith("diag"):
            ssm_args["P_scale"] = 0.0
        args = init[4:].split("-")
        assert args[0] == ""
        if len(args) > 1:
            ssm_args["init"] = args[1]
        A, P, B, V = dplr(N=N, rank=R, H=H, **ssm_args)
    else:
        A, P, B, V = hippo.nplr(init, N, R, **ssm_args)
        A = repeat(A, 'n -> s n', s=H)
        P = repeat(P, 'r n -> r s n', s=H)
        B = repeat(B, 'n -> s n', s=H)
        V = repeat(V, 'n m -> s n m', s=H)
    return A, P, B, V

combinations = {
    'hippo': ['legs', 'fourier'],
    'diag': ['diag-inv', 'diag-lin'],
    'all': ['legs', 'fourier', 'diag-inv', 'diag-lin'],
}

def combination(inits, N, R, S, **ssm_args):
    if isinstance(inits, str):
        inits = combinations[inits] if inits in combinations else [inits]

    assert S % len(inits) == 0, f"{S} independent trainable SSM copies must be multiple of {len(inits)} different inits"
    A, P, B, V = zip(
        *[ssm(init, N, R, S // len(inits), **ssm_args) for init in inits]
    )
    A = torch.cat(A, dim=0) # (S N)
    P = torch.cat(P, dim=1) # (R S N)
    B = torch.cat(B, dim=0) # (S N)
    V = torch.cat(V, dim=0) # (S N N)
    return A, P, B, V
