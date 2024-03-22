import math

import torch
import torch.nn.functional as F
import pytest

from einops import rearrange

from src.ops.vandermonde import log_vandermonde, log_vandermonde_fast


@pytest.mark.parametrize('L', [3, 17, 489, 2**10, 1047, 2**11, 2**12])
@pytest.mark.parametrize('N', [4, 8, 16, 32, 64, 128, 256])
# @pytest.mark.parametrize('L', [2048])
# @pytest.mark.parametrize('N', [64])
def test_vand_mult_symmetric(N, L):
    assert log_vandermonde_fast is not None, 'cauchy extension is not installed'
    rtol, atol = (1e-4, 1e-4) if N <= 64 and L <= 1024 else(1e-3, 1e-3)
    device = 'cuda'
    batch_size = 4
    torch.random.manual_seed(2357)
    v = torch.randn(batch_size, N // 2, dtype=torch.cfloat, device=device, requires_grad=True)
    x = (0.001 * torch.rand(batch_size, N // 2, device=device)
         + 1j * N * torch.rand(batch_size, N // 2, device=device))
    x.requires_grad_()
    v_keops = v.detach().clone().requires_grad_()
    x_keops = x.detach().clone().requires_grad_()
    out_keops = log_vandermonde(v_keops, x_keops, L)
    out = log_vandermonde_fast(v, x, L)
    err_out = (out - out_keops).abs()

    dout = torch.randn_like(out)
    dv_keops, dx_keops = torch.autograd.grad(out_keops, (v_keops, x_keops), dout, retain_graph=True)
    dv, dx = torch.autograd.grad(out, (v, x), dout, retain_graph=True)
    err_dv = (dv - dv_keops).abs()
    err_dx = (dx - dx_keops).abs()

    print(f'out error: max {err_out.amax().item():.6f}, mean {err_out.mean().item():.6f}')
    print(f'dv error: max {err_dv.amax().item():.6f}, mean {err_dv.mean().item():.6f}')
    print(f'dx relative error: max {err_dx.amax().item():.6f}, mean {err_dx.mean().item():.6f}')

    assert torch.allclose(out, out_keops, rtol=rtol, atol=atol)
    assert torch.allclose(dv, dv_keops, rtol=rtol, atol=atol)
    assert torch.allclose(dx, dx_keops, rtol=rtol, atol=atol)
