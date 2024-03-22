import math
import torch

import pytest

from einops import rearrange

from cauchy import cauchy_mult_torch, cauchy_mult_keops, cauchy_mult


def generate_data(batch_size, N, L, symmetric=True, device='cuda'):
    if not symmetric:
        v = torch.randn(batch_size, N, dtype=torch.complex64, device=device, requires_grad=True)
        w = torch.randn(batch_size, N, dtype=torch.complex64, device=device, requires_grad=True)
        z = torch.randn(L, dtype=torch.complex64, device=device)
    else:
        assert N % 2 == 0
        v_half = torch.randn(batch_size, N // 2, dtype=torch.complex64, device=device)
        v = torch.cat([v_half, v_half.conj()], dim=-1).requires_grad_(True)
        w_half = torch.randn(batch_size, N // 2, dtype=torch.complex64, device=device)
        w = torch.cat([w_half, w_half.conj()], dim=-1).requires_grad_(True)
        z = torch.exp(1j * torch.randn(L, dtype=torch.float32, device=device))
    return v, z, w


def grad_to_half_grad(dx):
    dx_half, dx_half_conj = dx.chunk(2, dim=-1)
    return dx_half + dx_half_conj.conj()


@pytest.mark.parametrize('L', [3, 17, 489, 2**10, 1047, 2**11, 2**12, 2**13, 2**14, 2**18])
@pytest.mark.parametrize('N', [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
def test_cauchy_mult_symmetric(N, L):
    # rtol, atol = (1e-4, 1e-4) if N <= 64 and L <= 1024 else(1e-3, 1e-3)
    atol = 1e-4
    tol_factor = 2.0  # Our error shouldn't be this much higher than Keops' error
    device = 'cuda'
    batch_size = 4
    torch.random.manual_seed(2357)
    v, z, w = generate_data(batch_size, N, L, symmetric=True, device=device)
    v_half = v[:, :N // 2].clone().detach().requires_grad_(True)
    w_half = w[:, :N // 2].clone().detach().requires_grad_(True)
    # out_torch = cauchy_mult_torch(v, z, w, symmetric=True)
    out_torch = cauchy_mult_torch(v.cdouble(), z.cdouble(), w.cdouble(), symmetric=True).cfloat()
    out_keops = cauchy_mult_keops(v, z, w)
    out = cauchy_mult(v_half, z, w_half)
    relerr_out_keops = (out_keops - out_torch).abs() / out_torch.abs()
    relerr_out = (out - out_torch).abs() / out_torch.abs()

    dout = torch.randn_like(out)
    dv_torch, dw_torch = torch.autograd.grad(out_torch, (v, w), dout, retain_graph=True)
    dv_torch, dw_torch = dv_torch[:, :N // 2], dw_torch[:, :N // 2]
    dv_keops, dw_keops = torch.autograd.grad(out_keops, (v, w), dout, retain_graph=True)
    dv_keops, dw_keops = grad_to_half_grad(dv_keops), grad_to_half_grad(dw_keops)
    dv, dw = torch.autograd.grad(out, (v_half, w_half), dout, retain_graph=True)
    relerr_dv_keops = (dv_keops - dv_torch).abs() / dv_torch.abs()
    relerr_dv = (dv - dv_torch).abs() / dv_torch.abs()
    relerr_dw_keops = (dw_keops - dw_torch).abs() / dw_torch.abs()
    relerr_dw = (dw - dw_torch).abs() / dw_torch.abs()
    print(f'Keops out relative error: max {relerr_out_keops.amax().item():.6f}, mean {relerr_out_keops.mean().item():6f}')
    print(f'out relative error: max {relerr_out.amax().item():.6f}, mean {relerr_out.mean().item():.6f}')
    print(f'Keops dv relative error: max {relerr_dv_keops.amax().item():.6f}, mean {relerr_dv_keops.mean().item():6f}')
    print(f'dv relative error: max {relerr_dv.amax().item():.6f}, mean {relerr_dv.mean().item():.6f}')
    print(f'Keops dw relative error: max {relerr_dw_keops.amax().item():.6f}, mean {relerr_dw_keops.mean().item():6f}')
    print(f'dw relative error: max {relerr_dw.amax().item():.6f}, mean {relerr_dw.mean().item():.6f}')
    assert (relerr_out.amax() <= relerr_out_keops.amax() * tol_factor + atol)
    assert (relerr_out.mean() <= relerr_out_keops.mean() * tol_factor + atol)
    # assert torch.allclose(out, out_torch, rtol=rtol, atol=atol)
    # assert torch.allclose(out, out_keops, rtol=rtol, atol=atol)
    assert (relerr_dv.amax() <= relerr_dv_keops.amax() * tol_factor + atol)
    assert (relerr_dv.mean() <= relerr_dv_keops.mean() * tol_factor + atol)
    assert (relerr_dw.amax() <= relerr_dw_keops.amax() * tol_factor + atol)
    assert (relerr_dw.mean() <= relerr_dw_keops.mean() * tol_factor + atol)
    # assert torch.allclose(dv, dv_torch, rtol=1e-4, atol=1e-4)
    # assert torch.allclose(dv, dv_keops, rtol=1e-4, atol=1e-4)
    # assert torch.allclose(dw, dw_torch, rtol=1e-4, atol=1e-4)
    # assert torch.allclose(dw, dw_keops, rtol=1e-4, atol=1e-4)
