# Modified by Mzero #20240123
# Copyright (C) 2023, Tri Dao.
import math
import torch
import torch.nn.functional as F
import pytest
from einops import rearrange
from selective_scan import selective_scan_fn, selective_scan_ref


@pytest.mark.parametrize('wtype', [torch.float32])
@pytest.mark.parametrize('itype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('seqlen', [64, 128, 256, 372, 512, 784, 1024, 1134, 2048, 4096])
@pytest.mark.parametrize('has_delta_bias', [False, True])
@pytest.mark.parametrize('delta_softplus', [False, True])
@pytest.mark.parametrize('has_D', [False, True])
@pytest.mark.parametrize("varBC_groups", [1, 2])
@pytest.mark.parametrize("nrows", [1, 4])
def test_selective_scan(varBC_groups, has_D, has_delta_bias,
                        delta_softplus, seqlen, itype, wtype, nrows):
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 8
    dstate = 8
    A = (-0.5 * torch.rand(dim, dstate, device=device, dtype=wtype)).requires_grad_()
    if varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen)
    B = torch.randn(*B_shape, device=device, dtype=itype,
                    requires_grad=True)
    if varBC_groups == 1:
        C_shape = (batch_size, dstate, seqlen)
    else:
        C_shape = (batch_size, varBC_groups, dstate, seqlen)
    C = torch.randn(*C_shape, device=device, dtype=itype,
                    requires_grad=True)
    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None
    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None
    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()
    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()
    D_ref = D.detach().clone().requires_grad_() if D is not None else None
    u_ref = u.detach().clone().requires_grad_()
    delta_ref = delta.detach().clone().requires_grad_()
    delta_bias_ref = delta_bias.detach().clone().requires_grad_() if delta_bias is not None else None
    out, *rest = selective_scan_fn(
        u, delta, A, B, C, D,
        delta_bias=delta_bias, delta_softplus=delta_softplus, nrows=nrows,
    )
    out_ref, *rest = selective_scan_ref(
        u_ref, delta_ref, A_ref, B_ref, C_ref, D_ref,
        delta_bias=delta_bias_ref, delta_softplus=delta_softplus,
    )
    # dA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # dt_u = delta * u

    print(f'Output max diff: {(out - out_ref).abs().max().item()}')
    print(f'Output mean diff: {(out - out_ref).abs().mean().item()}')
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol)

    g = torch.randn_like(out)
    out_ref.backward(g)
    out.backward(g)

    print(f'du max diff: {(u.grad - u_ref.grad).abs().max().item()}')
    print(f'ddelta max diff: {(delta.grad - delta_ref.grad).abs().max().item()}')
    print(f'dA max diff: {(A.grad - A_ref.grad).abs().max().item()}')
    print(f'dB max diff: {(B.grad - B_ref.grad).abs().max().item()}')
    print(f'dC max diff: {(C.grad - C_ref.grad).abs().max().item()}')
    if has_D:
        print(f'dD max diff: {(D.grad - D_ref.grad).abs().max().item()}')
    if has_delta_bias:
        print(f'ddelta_bias max diff: {(delta_bias.grad - delta_bias_ref.grad).abs().max().item()}')

    assert torch.allclose(u.grad, u_ref.grad.to(dtype=itype), rtol=rtol * 2, atol=atol * 2)
    assert torch.allclose(delta.grad, delta_ref.grad.to(dtype=itype), rtol=rtol * 5, atol=atol * 10)
    assert torch.allclose(A.grad, A_ref.grad, rtol=rtolw, atol=atolw * 5)
    assert torch.allclose(B.grad, B_ref.grad, rtol=rtol, atol=atol)
    assert torch.allclose(C.grad, C_ref.grad, rtol=rtol, atol=atol)
    if has_D:
        assert torch.allclose(D.grad, D_ref.grad, rtol=rtolw, atol=atolw)
    if has_delta_bias:
        assert torch.allclose(delta_bias.grad, delta_bias_ref.grad, rtol=rtolw, atol=atolw)

