if True:
    import math
    import torch
    class PScan(torch.autograd.Function):
        @staticmethod
        def pscan(A, X):
            # A : (B, D, L, N)
            # X : (B, D, L, N)

            # modifies X in place by doing a parallel scan.
            # more formally, X will be populated by these values :
            # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
            # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)
            
            B, D, L, _ = A.size()
            num_steps = int(math.log2(L))

            # up sweep or reduction step
            Aa = A
            Xa = X
            for k in range(num_steps):
                T = 2 * (Xa.size(2) // 2)

                Aa = Aa[:, :, :T].view(B, D, T//2, 2, -1)
                Xa = Xa[:, :, :T].view(B, D, T//2, 2, -1)
                
                Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
                Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

                Aa = Aa[:, :, :, 1]
                Xa = Xa[:, :, :, 1]

            # down sweep
            for k in range(num_steps-1, -1, -1):
                Aa = A[:, :, 2**k-1:L:2**k]
                Xa = X[:, :, 2**k-1:L:2**k]

                T = 2 * (Xa.size(2) // 2)

                if T < Xa.size(2):
                    Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                    Aa[:, :, -1].mul_(Aa[:, :, -2])

                Aa = Aa[:, :, :T].view(B, D, T//2, 2, -1)
                Xa = Xa[:, :, :T].view(B, D, T//2, 2, -1)

                Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
                Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

        @staticmethod
        def forward(ctx, A_in, X_in):
            """
            Applies the parallel scan operation, as defined above. Returns a new tensor.

            Args:
                A_in : (B, L, D, N)
                X_in : (B, L, D, N)

            Returns:
                H : (B, L, D, N)
            """

            # clone tensor (in-place ops)
            A = A_in.clone() # (B, L, D, N)
            X = X_in.clone() # (B, L, D, N)
            
            # prepare tensors
            A = A.transpose(2, 1) # (B, D, L, N)
            X = X.transpose(2, 1) # (B, D, L, N)

            # parallel scan
            PScan.pscan(A, X)

            ctx.save_for_backward(A_in, X)

            return X.transpose(2, 1)
        
        @staticmethod
        def backward(ctx, grad_output_in):
            """
            Flows the gradient from the output to the input. Returns two new tensors.

            Args:
                ctx : A_in : (B, L, D, N), X : (B, D, L, N)
                grad_output_in : (B, L, D, N)

            Returns:
                gradA : (B, L, D, N), gradX : (B, L, D, N)
            """

            A_in, X = ctx.saved_tensors

            # clone tensors 
            A = A_in.clone()
            # grad_output_in will be cloned with flip()

            # prepare tensors
            A = A.transpose(2, 1) # (B, D, L, N)
            A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
            grad_output_b = grad_output_in.transpose(2, 1)

            # reverse parallel scan
            grad_output_b = grad_output_b.flip(2)
            PScan.pscan(A, grad_output_b)
            grad_output_b = grad_output_b.flip(2)

            Q = torch.zeros_like(X)
            Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

            return Q.transpose(2, 1), grad_output_b.transpose(2, 1)
        
    pscan = PScan.apply

    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
    from nrow_selective_scan import selective_scan_fn as selective_scan_fn_v2


def pscan_selective_scan(
    x, 
    A, 
    D, 
    x_proj_weight = None,
    x_proj_bias = None,
    dt_proj_weight = None,
    dt_proj_bias = None,
):
    """
    x (B, H, W, D)
    A (K, D, N)
    D (K, D)
    y (B, L, D)  
    x_proj_weight = None # K, (R + N + N), D
    x_proj_bias = None
    dt_proj_weight = None # K, D, R
    dt_proj_bias = None # K, D
    """
    
    _, d_inner, d_state = A.shape
    _B, _H, _W, d_inner = x.shape
    x = x.view(_B, -1, d_inner)
    dt_rank = d_inner // 16

    xs = torch.stack([
        x, 
        x.view(_B, _H, _W, d_inner).transpose(dim0=1, dim1=2).contiguous().view(_B, -1, d_inner)
    ], dim=2) # (B, L, 2, D)
    xs = torch.cat([xs, torch.flip(xs, dims=[1])], dim=2) # (B, L, 4, D)
    xdbcs = torch.einsum("blkd,kcd->blkc", xs, x_proj_weight)  # (B, L, 4, C)
    dts, Bs, Cs = torch.split(xdbcs, [dt_rank, d_state, d_state], dim=3) # (B, L, 4, C)
    dts = torch.einsum("blkr,kdr->blkd", dts, dt_proj_weight)
    dts = dts + dt_proj_bias
    dtAs = torch.exp(dts.unsqueeze(4) * A) # (B, L, 4, D, N)
    dtBs = dts.unsqueeze(4) * Bs.unsqueeze(3) # (B, L, 4, D, N)
    dtBxs = dtBs * xs.unsqueeze(4) # (B, L, 4, D, N)
    hs = pscan(dtAs.flatten(2, 3), dtBxs.flatten(2, 3)).view(_B, -1, 4, d_inner, d_state)
    y = torch.einsum("blkdn,blkn->blkd", hs, Cs)
    y = (y + D * xs).sum(dim=2)
    return y, dtAs.sum(dim=2), dtBxs.sum(dim=2), xdbcs.sum(dim=2), xs.sum(dim=2), dts.sum(dim=2)


def pscan_selective_scan_share(
    x, 
    A, 
    D, 
    x_proj_weight = None,
    x_proj_bias = None,
    dt_proj_weight = None,
    dt_proj_bias = None,
):
    """
    x (B, H, W, D)
    A (D, N)
    D (D)
    y (B, L, D)  
    x_proj_weight = None # (R + N + N), D
    x_proj_bias = None
    dt_proj_weight = None # D, R
    dt_proj_bias = None # D
    """
    d_inner, d_state = A.shape
    _B, _H, _W, d_inner = x.shape
    x = x.view(_B, -1, d_inner)
    dt_rank = d_inner // 16

    xdbc = torch.einsum("bld,cd->blc", x, x_proj_weight)  # (B, L, C)
    dt, B, C = torch.split(xdbc, [dt_rank, d_state, d_state], dim=2) # (B, L, C)
    dt = torch.einsum("blr,dr->bld", dt, dt_proj_weight)
    dt = dt + dt_proj_bias
    dtA = torch.exp(dt.unsqueeze(3) * A) # (B, L, D, N)
    dtB = dt.unsqueeze(3) * B.unsqueeze(2) # (B, L, D, N)
    dtBx = dtB * x.unsqueeze(3) # (B, L, D, N)

    dtAdtBxs = torch.stack([dtA, dtBx], dim=2) # (B, L, 2, D, N)
    dtAdtBxs = torch.stack([
        dtAdtBxs, 
        dtAdtBxs.view(_B, _H, _W, 2, d_inner, d_state).transpose(dim0=1, dim1=2).contiguous().flatten(1, 2)
    ], dim=3) # (B, L, 2, 2, D, N)
    dtAdtBxs = torch.cat([dtAdtBxs, torch.flip(dtAdtBxs, dims=[1])], dim=3) # (B, L, 2, 4, D, N)
    hs = pscan(
        dtAdtBxs[:, :, 0].contiguous().view(_B, -1, 4 * d_inner, d_state),
        dtAdtBxs[:, :, 1].contiguous().view(_B, -1, 4 * d_inner, d_state),
    ).view(_B, -1, 4, d_inner, d_state).sum(dim=2) # (B, L, D, N)
    y = torch.einsum("bldn,bln->bld", hs, C)
    y = y + D * x
    return y, dtAdtBxs[:, :, 0].sum(dim=2), dtAdtBxs[:, :, 1].sum(dim=2), 


def pscan_selective_scan_share_exp(
    x, 
    A, 
    D, 
    x_proj_weight = None,
    x_proj_bias = None,
    dt_proj_weight = None,
    dt_proj_bias = None,
):
    """
    x (B, H, W, D)
    A (D, N)
    D (D)
    y (B, L, D)  
    x_proj_weight = None # (R + N + N), D
    x_proj_bias = None
    dt_proj_weight = None # D, R
    dt_proj_bias = None # D
    """
    d_inner, d_state = A.shape
    _B, _H, _W, d_inner = x.shape
    x = x.view(_B, -1, d_inner)
    dt_rank = d_inner // 16

    xs = torch.stack([
        x, 
        x.view(_B, _H, _W, d_inner).transpose(dim0=1, dim1=2).contiguous().view(_B, -1, d_inner)
    ], dim=2) # (B, L, 2, D)
    xs = torch.cat([xs, torch.flip(xs, dims=[1])], dim=2) # (B, L, 4, D)
    xs = xs.sum(dim=2)

    xdbc = torch.einsum("bld,cd->blc", xs, x_proj_weight)  # (B, L, C)
    dt, B, C = torch.split(xdbc, [dt_rank, d_state, d_state], dim=2) # (B, L, C)
    dt = torch.einsum("blr,dr->bld", dt, dt_proj_weight)
    dt = dt + dt_proj_bias
    dtA = torch.exp(dt.unsqueeze(3) * A) # (B, L, D, N)
    dtB = dt.unsqueeze(3) * B.unsqueeze(2) # (B, L, D, N)
    dtBx = dtB * x.unsqueeze(3) # (B, L, D, N)
    
    hs = pscan(
        dtA,
        dtBx,
    ) # (B, L, D, N)
    y = torch.einsum("bldn,bln->bld", hs, C)

    y = y + D * x
    return y, dtA, dtBx, xdbc, xs, dt


def pscan_selective_scan_cuda(
    x, 
    A, 
    D, 
    x_proj_weight = None,
    x_proj_bias = None,
    dt_proj_weight = None,
    dt_proj_bias = None,
    nrows=1,
):
    """
    x (B, H, W, D)
    A (K, D, N)
    D (K, D)
    y (B, L, D)  
    x_proj_weight = None # K, (R + N + N), D
    x_proj_bias = None
    dt_proj_weight = None # K, D, R
    dt_proj_bias = None # K, D
    """
    
    _, d_inner, d_state = A.shape
    _B, _H, _W, d_inner = x.shape
    x = x.permute(0, 3, 1, 2)
    dt_rank = d_inner // 16

    if True:
        def cross_scan_2d(x):
            # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
            x_hwwh = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
            xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
            return xs
            
        if True:
            xs = cross_scan_2d(x) # (b, k, d, l)

            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_proj_weight)

            xs = xs.flatten(1, 2) # (b, k * d, l)
            dts = dts.contiguous().flatten(1, 2) # (b, k * d, l)
            As = -torch.exp(A.float()).flatten(0, 1)  # (k * d, d_state)
            Ds = D.flatten(0, 1) # (k * d)
            dt_projs_bias = dt_proj_bias.flatten(0, 1) # (k * d)

            # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
            # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
            # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

            out_y = selective_scan_fn_v1(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                nrows=nrows,
            ).view(_B, 4, d_inner, -1)
            # assert out_y.dtype == torch.float16

    y = out_y.sum(dim=1)

    return y.permute(0, 2, 1)


def test_s6():
    from mamba_ssm import selective_scan_fn
    device = torch.device("cuda")
    dtype = torch.float32
    B, L, G, D, N, R = 3, 4096, 4, 192, 16, 192 // 16
    # B, L, G, D, N, R = 3, 4096, 1, 192, 16, 192 // 16
    xi = torch.randn((B, G * D, L), device=device, dtype=dtype)
    Ai = torch.randn((G * D, N), device=device, dtype=dtype)
    Di = torch.randn((G * D), device=device, dtype=dtype)
    dti = torch.randn((B, G * D, L), device=device, dtype=dtype)
    Bi = torch.randn((B, G, N, L), device=device, dtype=dtype)
    Ci = torch.randn((B, G, N, L), device=device, dtype=dtype)
    tpb = torch.randn((G * D), device=device, dtype=dtype)

    Ai2 = torch.randn((G * D, 4*N), device=device, dtype=dtype)
    Bi2 = torch.randn((B, G, 4*N, L), device=device, dtype=dtype)
    Ci2 = torch.randn((B, G, 4*N, L), device=device, dtype=dtype)

    selective_scan_fn(xi, dti, Ai, Bi, Ci, Di, None, tpb, True, False)

    import time
    tim0 = time.time()
    for _ in range(1000):
        y = selective_scan_fn(xi, dti, Ai, Bi, Ci, Di, None, tpb, True, False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    tim1 = time.time()
    for _ in range(1000):
        y = selective_scan_fn(xi, dti, Ai2, Bi2, Ci2, Di, None, tpb, True, False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    tim2 = time.time()
    print(tim1-tim0, tim2-tim1, torch.cuda.max_memory_allocated()) # 0.7172577381134033 2.400775194168091 185063424
    time.sleep(100000)

def test_mamba():
    from mamba_ssm import mamba_inner_fn
    device = torch.device("cuda")
    dtype = torch.float32
    B, L, G, D, N, R, W = 3, 4096, 1, 192, 16, 192 // 16, 4
    xzi = torch.randn((B, 2 * D, L), device=device, dtype=dtype)
    cv1dw = torch.randn((D, 1, W), device=device, dtype=dtype)
    cv1db = torch.randn((D), device=device, dtype=dtype)
    xpw = torch.randn(((R + N + N), D), device=device, dtype=dtype)
    xpb = torch.randn(((R + N + N)), device=device, dtype=dtype)
    tpw = torch.randn((D, R), device=device, dtype=dtype)
    tpb = torch.randn((D), device=device, dtype=dtype)
    opw = torch.randn((D // 2, D), device=device, dtype=dtype)
    opb = torch.randn((D // 2), device=device, dtype=dtype)

    Ai = torch.randn((D, N), device=device, dtype=dtype)
    Di = torch.randn((D), device=device, dtype=dtype)
    tpb = torch.randn((G * D), device=device, dtype=dtype)

    xpw2 = torch.randn(((R + 4*N + 4*N), D), device=device, dtype=dtype)
    Ai2 = torch.randn((D, 4*N), device=device, dtype=dtype)
    xzi2 = xzi.clone().detach()
    cv1dw2 = cv1dw.clone().detach()
    cv1db2 = cv1db.clone().detach()
    tpw2 = tpw.clone().detach()
    tpb2 = tpb.clone().detach()
    opw2 = opw.clone().detach()
    opb2 = opb.clone().detach()
    Di2 = Di.clone().detach()
    tpb2 = tpb.clone().detach()

    # the first time would be slower
    mamba_inner_fn(xzi, cv1dw, cv1db, xpw, tpw, opw, opb, Ai, None, None, Di, tpb, None, None, True)

    import time
    tim0 = time.time()
    for _ in range(1000):
        y = mamba_inner_fn(xzi, cv1dw, cv1db, xpw, tpw, opw, opb, Ai, None, None, Di, tpb, None, None, True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    tim1 = time.time()
    for _ in range(1000):
        y = mamba_inner_fn(xzi2, cv1dw2, cv1db2, xpw2, tpw2, opw2, opb2, Ai2, None, None, Di2, tpb2, None, None, True)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    tim2 = time.time()
    print(tim1-tim0, tim2-tim1, torch.cuda.max_memory_allocated()) # 0.3865385055541992 1.0227293968200684 9436108
    time.sleep(100000)


def test_mamba_a(d_state=16):
    import torch
    import time

    from mamba_ssm import Mamba

    batch, length, dim = 1, 4000, 1024
    # d_state = 128

    torch.manual_seed(1)

    model = Mamba(
        d_model=dim, # Model dimension d_model
        d_state=d_state,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    start_time = time.time()
    N = 500
    for _ in range(N):
        x = torch.randn(batch, length, dim).to("cuda")
        y_cuda = model(x)
        loss = torch.norm(y_cuda)

        optim.zero_grad()
        loss.backward()
        optim.step()

    end_time = time.time()

    res = (end_time-start_time)/N
    print(f"B={batch}, L={length}, d_model={dim}, d_state={model.d_state}, time (ms)={res*1000}")

def test_mamba_(d_state=16):
    import torch
    import time

    from mamba_ssm import Mamba

    batch, length, dim = 1, 4000, 1024
    # d_state = 128

    torch.manual_seed(1)

    model = Mamba(
        d_model=dim, # Model dimension d_model
        d_state=d_state,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    start_time = time.time()
    N = 500
    for _ in range(N):
        x = torch.randn(batch, length, dim, device=torch.device("cuda"))
        y_cuda = model(x)
        loss = torch.norm(y_cuda)

        optim.zero_grad()
        loss.backward()
        optim.step()

    end_time = time.time()

    res = (end_time-start_time)/N
    print(f"B={batch}, L={length}, d_model={dim}, d_state={model.d_state}, time (ms)={res*1000}")

def test_mamba2_(d_state=16, batch=1, N=500):
    import torch
    import time

    from mamba_ssm import Mamba

    # batch, length, dim = 1, 4000, 1024
    # d_state = 128
    length, dim = 4000, 1024

    torch.manual_seed(1)

    model = Mamba(
        d_model=dim, # Model dimension d_model
        d_state=d_state,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")

    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)
    # x = torch.randn(batch, length, dim, device=torch.device("cuda"))

    start_time = time.time()
    # N = 500
    data = torch.randn(batch, length, dim)
    for _ in range(N):
        x = data.to('cuda', non_blocking=True)
        y_cuda = model(x)
        loss = torch.norm(y_cuda)

        optim.zero_grad()
        loss.backward()
        optim.step()

    end_time = time.time()

    res = (end_time-start_time)/N
    print(f"B={batch}, L={length}, d_model={dim}, d_state={model.d_state}, time (ms)={res*1000}")


def test():
    device = torch.device("cuda")
    dtype = torch.float32
    B, H, W, D, N, R = 3, 56, 56, 192, 16, 192 // 16
    xi = torch.randn((B, H, W, D), device=device, dtype=dtype)
    Ai = torch.randn((4, D, N), device=device, dtype=dtype)
    Di = torch.randn((4, D), device=device, dtype=dtype)
    xpw = torch.randn((4, (R + N + N), D), device=device, dtype=dtype)
    xpb = torch.randn((4, (R + N + N)), device=device, dtype=dtype)
    tpw = torch.randn((4, D, R), device=device, dtype=dtype)
    tpb = torch.randn((4, D), device=device, dtype=dtype)

    N = 128
    Ai2 = torch.randn((4, D, N), device=device, dtype=dtype)
    xpw2 = torch.randn((4, (R + N + N), D), device=device, dtype=dtype)
    xpb2 = torch.randn((4, (R + N + N)), device=device, dtype=dtype)

    res0 = pscan_selective_scan_cuda(xi, Ai, Di, xpw, xpb, tpw, tpb, nrows=1)
    res1 = pscan_selective_scan_cuda(xi, Ai, Di, xpw, xpb, tpw, tpb, nrows=8)
    print((res0 - res1).abs().sum())

    import time
    tim0 = time.time()
    for _ in range(1000):
        y = pscan_selective_scan_cuda(xi, Ai, Di, xpw, xpb, tpw, tpb, nrows=1)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # memory: 1773864448
    tim1 = time.time()
    for _ in range(1000):
        y = pscan_selective_scan_cuda(xi, Ai2, Di, xpw2, xpb2, tpw, tpb, nrows=1)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # memory: 1691860992
    tim2 = time.time()
    print(tim1-tim0, tim2-tim1, torch.cuda.max_memory_allocated())
    time.sleep(100000)

    xpw = xpw[0].unsqueeze(0).repeat(4, 1, 1)
    tpw = tpw[0].unsqueeze(0).repeat(4, 1, 1)
    xpb = xpb[0].unsqueeze(0).repeat(4, 1)
    tpb = tpb[0].unsqueeze(0).repeat(4, 1) * 0

    res0 = pscan_selective_scan_cuda(xi, Ai, Di, xpw, xpb, tpw, tpb)
    res1, da1, db1, dtbc1, xs1, dt1 = pscan_selective_scan(xi, Ai, Di, xpw, xpb, tpw, tpb)
    res2, da2, db2 = pscan_selective_scan_share(xi, Ai[0], Di[0], xpw[0], xpb[0], tpw[0], tpb[0])
    res3, da3, db3, dtbc3, xs3, dt3 = pscan_selective_scan_share_exp(xi, Ai[0], Di[0], xpw[0], xpb[0], tpw[0], tpb[0])

    # print(res0.shape, res1.shape, res0 - res1)
    # print(res2.shape, res3.shape, res2 - res3)
    print((xs1 - xs3).abs().mean())
    print((dtbc1 - dtbc3).abs().mean())
    print((dt1 - dt3).abs().mean())
    print(da1 - da3)
    print((da1 - da3).abs().mean())
    import time; time.sleep(10000)
    print(da2 - da3)
    print(db2 - db3)

    import time
    tim0 = time.time()
    for _ in range(100):
        y = pscan_selective_scan(xi, Ai, Di, xpw, xpb, tpw, tpb)
    torch.cuda.synchronize()
    # memory: 1773864448
    tim1 = time.time()
    for _ in range(100):
        y = pscan_selective_scan_share(xi, Ai[0], Di[0], xpw[0], xpb[0], tpw[0], tpb[0])
    torch.cuda.synchronize()
    # memory: 1691860992
    tim2 = time.time()
    for _ in range(100):
        y = pscan_selective_scan_cuda(xi, Ai, Di, xpw, xpb, tpw, tpb)
    torch.cuda.synchronize()
    # memory: 57894400
    tim3 = time.time()
    print(tim1-tim0, tim2-tim1, tim3-tim2, torch.cuda.max_memory_allocated())

# test_s6()
# test_mamba()

# test_mamba_a(16) # 34.6ms in 4090 # 31ms
# test_mamba_a(64) # 34ms in 4090 # 30.8ms
# test_mamba_a(128) # 45.3ms in 4090 # 43.1ms

# test_mamba_(16) # 17.4ms in 4090 # 17.0ms
# test_mamba_(64) # 28ms in 4090 # 27.3 ms
# test_mamba_(128) # 41.8ms in 4090 # 40.9ms

# test_mamba2_(16) # 18.54ms in 4090
# test_mamba2_(64) # 30ms in 4090
# test_mamba2_(128) # 44.4ms in 4090

# test_mamba2_(16, 48, 100) # 348ms in 4090 # 21G
# test_mamba2_(64, 48, 100) # 638ms in 4090; # 21G 
# test_mamba2_(128, 48, 100) # 1028ms in 4090; 22G

