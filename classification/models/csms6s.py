import time
import torch
import warnings


WITH_SELECTIVESCAN_OFLEX = True
WITH_SELECTIVESCAN_CORE = False
WITH_SELECTIVESCAN_MAMBA = True
try:
    import selective_scan_cuda_oflex
except ImportError:
    WITH_SELECTIVESCAN_OFLEX = False
    warnings.warn("Can not import selective_scan_cuda_oflex. This affects speed.")
    print("Can not import selective_scan_cuda_oflex. This affects speed.", flush=True)
try:
    import selective_scan_cuda_core
except ImportError:
    WITH_SELECTIVESCAN_CORE = False
try:
    import selective_scan_cuda
except ImportError:
    WITH_SELECTIVESCAN_MAMBA = False


def selective_scan_torch(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True, 
    *args,
    **kwargs
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
            
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    
    if True:
        x = A.new_zeros((Batch, KCdim, N))
        ys = []
        for i in range(L):
            x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
            y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (B, C, L)
    
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True, backend=None):
        ctx.delta_softplus = delta_softplus
        backend = "oflex" if WITH_SELECTIVESCAN_OFLEX and (backend is None) else backend
        backend = "core" if WITH_SELECTIVESCAN_CORE and (backend is None) else backend
        backend = "mamba" if WITH_SELECTIVESCAN_MAMBA and (backend is None) else backend
        ctx.backend = backend
        if backend == "oflex":
            out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        elif backend == "core":
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        elif backend == "mamba":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        backend = ctx.backend
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if backend == "oflex":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "core":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )
        elif backend == "mamba":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
                False
            )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(
    u: torch.Tensor, # (B, K * C, L)
    delta: torch.Tensor, # (B, K * C, L)
    A: torch.Tensor, # (K * C, N)
    B: torch.Tensor, # (B, K, N, L)
    C: torch.Tensor, # (B, K, N, L)
    D: torch.Tensor = None, # (K * C)
    delta_bias: torch.Tensor = None, # (K * C)
    delta_softplus=True, 
    oflex=True,
    backend=None,
):
    WITH_CUDA = (WITH_SELECTIVESCAN_OFLEX or WITH_SELECTIVESCAN_CORE or WITH_SELECTIVESCAN_MAMBA)
    fn = selective_scan_torch if backend == "torch" or (not WITH_CUDA) else SelectiveScanCuda.apply
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


# fvcore flops =======================================
def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    import numpy as np
    
    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop
    

    assert not with_complex

    flops = 0 # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops

def selective_scan_flop_jit(inputs, outputs, backend="prefixsum", verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    flops_fn = flops_selective_scan_ref if backend == "naive" else flops_selective_scan_fn
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


if __name__ == "__main__":
    def params(B, K, C, N, L, device = torch.device("cuda"), itype = torch.float):
        As = (-0.5 * torch.rand(K * C, N, device=device, dtype=torch.float32)).requires_grad_()
        Bs = torch.randn((B, K, N, L), device=device, dtype=itype).requires_grad_()
        Cs = torch.randn((B, K, N, L), device=device, dtype=itype).requires_grad_()
        Ds = torch.randn((K * C), device=device, dtype=torch.float32).requires_grad_()
        u = torch.randn((B, K * C, L), device=device, dtype=itype).requires_grad_()
        delta = (0.5 * torch.rand((B, K * C, L),  device=device, dtype=itype)).requires_grad_()
        delta_bias = (0.5 * torch.rand((K * C), device=device, dtype=torch.float32)).requires_grad_()
        return u, delta, As, Bs, Cs, Ds, delta_bias

    def bench(func, xs, Warmup=30, NTimes=20):
        import time
        torch.cuda.synchronize()
        for r in range(Warmup):
            for x in xs:
                func(x)
        torch.cuda.synchronize()
        tim0 = time.time()
        for r in range(NTimes):
            for x in xs:
                func(x)
        torch.cuda.synchronize()
        return (time.time() - tim0) / NTimes

    def check():
        u, delta, As, Bs, Cs, Ds, delta_bias = params(1, 4, 16, 8, 512, itype=torch.float16)
        u1, delta1, As1, Bs1, Cs1, Ds1, delta_bias1 = [x.clone().detach().requires_grad_() for x in [u, delta, As, Bs, Cs, Ds, delta_bias]]
        
        # out_ref = selective_scan_fn(u, delta, As, Bs, Cs, Ds, delta_bias, True, backend="torch")
        out = selective_scan_fn(u1, delta1, As1, Bs1, Cs1, Ds1, delta_bias1, True, backend="oflex")
        out_ref = selective_scan_fn(u, delta, As, Bs, Cs, Ds, delta_bias, True, backend="mamba")
        print((out_ref - out).abs().max())
        out.sum().backward()
        out_ref.sum().backward()
        for x, y in zip([u, As, Bs, Cs, Ds, delta, delta_bias], [u1, As1, Bs1, Cs1, Ds1, delta1, delta_bias1]):
            print((x.grad - y.grad).abs().max())

        u, delta, As, Bs, Cs, Ds, delta_bias = params(128, 4, 96, 8, 56 * 56)
        print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="oflex"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))
        print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="mamba"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))
        print(bench(lambda x: selective_scan_fn(x[0], x[1], x[2], x[3], x[4], x[5], x[6], True, backend="torch"), [(u, delta, As, Bs, Cs, Ds, delta_bias),]))

    check()

