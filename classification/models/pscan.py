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

def test_mamba_slow(d_state=16, batch=1, N=500):
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
        use_fast_path=False,
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


def test_cross_scan():
    def cross_scan(x: torch.Tensor, dim=1, channel_first=True):
        # (B, C, H, W) => (B, K, C, H * W) with K = len([HW, WH, FHW, FWH])
        dim0, dim1, dim2 = 2, 3, 3
        # assert channel_first and dim < 3
        xs = torch.stack([x.flatten(dim0, dim1), x.transpose(dim0=dim0, dim1=dim1).contiguous().flatten(dim0, dim1)], dim=dim)
        xs = torch.cat([xs, torch.flip(xs, dims=[dim2])], dim=dim) # (b, k, d, l)
        return xs
        
    def cross_merge(x: torch.Tensor, dim=1, channel_first=True, shape=(1, 1)):
        # (B, K, D, L) => (B, D, H * W) with K = len([HW, WH, FHW, FWH])
        # assert channel_first and dim == 1
        B, K, D, L = x.shape
        fhwwh = torch.flip(x[:, 2:, :, :], dims=[3])
        hwwh = x[:, :2, :, :] + fhwwh
        wh = hwwh[:, 1, :, :].view(B, D, shape[1], shape[0]).transpose(dim0=2, dim1=3).contiguous()
        hw = hwwh[:, 1, :, :].view(B, D, shape[0], shape[1]) + wh
        return hw

    class crossScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, x: torch.Tensor):
            ctx.shape = x.shape # (B, C, H, W)
            return cross_scan(x)
        
        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, out):
            # out: (b, k, d, l)
            return cross_merge(out, shape=ctx.shape[-2:])
    
    cross_scan_fn = crossScan.apply

    device = torch.device("cuda")
    dtype = torch.float32
    B, H, W, D = 3, 56, 56, 192
    xi = torch.randn((B, D, H, W), device=device, dtype=dtype).requires_grad_()
    y = cross_scan_fn(xi)
    y = cross_scan(xi)
    
    import time
    tim0 = time.time()
    for _ in range(1000):
        y = cross_scan_fn(xi)
        y.mean().backward()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # memory: 1773864448
    tim1 = time.time()
    for _ in range(1000):
        y = cross_scan(xi)
        y.mean().backward()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # memory: 1691860992
    tim2 = time.time()
    print(tim1-tim0, tim2-tim1, torch.cuda.max_memory_allocated())
    time.sleep(100000)


if True:
    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat
    import selective_scan_cuda_core as selective_scan_cuda

    class SelectiveScanFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            # input_t: float, fp16, bf16; weight_t: float;
            # u, B, C, delta: input_t
            # D, delta_bias: float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True
            if D is not None and (D.dtype != torch.float):
                ctx._d_dtype = D.dtype
                D = D.float()
            if delta_bias is not None and (delta_bias.dtype != torch.float):
                ctx._delta_bias_dtype = delta_bias.dtype
                delta_bias = delta_bias.float()
            
            assert u.shape[1] % (B.shape[1] * nrows) == 0 
            assert nrows in [1, 2, 4] # 8+ is too slow to compile

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            
            _dD = None
            if D is not None:
                if dD.dtype != getattr(ctx, "_d_dtype", dD.dtype):
                    _dD = dD.to(ctx._d_dtype)
                else:
                    _dD = dD

            _ddelta_bias = None
            if delta_bias is not None:
                if ddelta_bias.dtype != getattr(ctx, "_delta_bias_dtype", ddelta_bias.dtype):
                    _ddelta_bias = ddelta_bias.to(ctx._delta_bias_dtype)
                else:
                    _ddelta_bias = ddelta_bias

            return (du, ddelta, dA, dB, dC, _dD, _ddelta_bias, None, None)

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        return SelectiveScanFn.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    def forward_corev1(
        self, 
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...

        B, C, H, W = x.shape
        L = H * W
        D, d_state = A_logs.shape
        K, D, dt_rank = dt_projs_weight.shape

        xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(A_logs.to(torch.float))  # (k * d, d_state)
        Ds = Ds.to(torch.float) # (k * d)
        dt_projs_bias = dt_projs_bias.to(torch.float).view(-1) # (k * d)
        ys: torch.Tensor = selective_scan(
                xs.to(torch.float), 
                dts.to(torch.float), 
                As, 
                Bs.to(torch.float), 
                Cs.to(torch.float), 
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                nrows=nrows,
            ).view(B, 4, -1, L)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y).to(x.dtype)
        
        return y

    def crossmerge(
        self, 
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...

        B, C, H, W = x.shape
        L = H * W
        D, d_state = A_logs.shape
        K, D, dt_rank = dt_projs_weight.shape

        xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)

        # x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        # if x_proj_bias is not None:
        #     x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        # dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
        # dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        # xs = xs.view(B, -1, L) # (b, k * d, l)
        # dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        # As = -torch.exp(A_logs.to(torch.float))  # (k * d, d_state)
        # Ds = Ds.to(torch.float) # (k * d)
        # dt_projs_bias = dt_projs_bias.to(torch.float).view(-1) # (k * d)
        # ys: torch.Tensor = selective_scan(
        #         xs.to(torch.float), 
        #         dts.to(torch.float), 
        #         As, 
        #         Bs.to(torch.float), 
        #         Cs.to(torch.float), 
        #         Ds,
        #         delta_bias=dt_projs_bias,
        #         delta_softplus=True,
        #         nrows=nrows,
        #     ).view(B, 4, -1, L)
        
        ys = xs
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # y = out_norm(y).to(x.dtype)
        
        return y

    def linears(
        self, 
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        # float32 should be true in training!!!! otherwise, the output of selective_scan would be inf...

        B, C, H, W = x.shape
        L = H * W
        D, d_state = A_logs.shape
        K, D, dt_rank = dt_projs_weight.shape

        # xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).contiguous().flatten(2, 3)], dim=1)
        # xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)
        xs = x.view(B, 1, -1, L).repeat(1, 4, 1, 1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        # xs = xs.view(B, -1, L) # (b, k * d, l)
        # dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        # As = -torch.exp(A_logs.to(torch.float))  # (k * d, d_state)
        # Ds = Ds.to(torch.float) # (k * d)
        # dt_projs_bias = dt_projs_bias.to(torch.float).view(-1) # (k * d)
        # ys: torch.Tensor = selective_scan(
        #         xs.to(torch.float), 
        #         dts.to(torch.float), 
        #         As, 
        #         Bs.to(torch.float), 
        #         Cs.to(torch.float), 
        #         Ds,
        #         delta_bias=dt_projs_bias,
        #         delta_softplus=True,
        #         nrows=nrows,
        #     ).view(B, 4, -1, L)
        
        # ys = xs
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)

        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        # y = out_norm(y).to(x.dtype)
        
        return dts.sum() + Bs.sum() + Cs.sum()

    class crossScan1(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, x: torch.Tensor):
            ctx.shape = x.shape # (B, C, H, W)
            xs = torch.stack([x.flatten(2, 3), x.transpose(dim0=2, dim1=3).flatten(2, 3)], dim=1)
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)
            return xs
        
        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, K, D, L = ys.shape
            B, C, H, W = ctx.shape
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    
    class crossMerge1(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, ys: torch.Tensor, H, W):
            B, K, D, L = ys.shape
            ctx.shape = (H, W)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y
        
        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            x1 = x.view(*x.shape[:2], *ctx.shape) # (B, D, H, W)
            xs = torch.stack([x, x1.transpose(dim0=2, dim1=3).flatten(2, 3)], dim=1)
            xs = torch.cat([xs, torch.flip(xs, dims=[-1])], dim=1) # (b, k, d, l)
            return xs, None, None
    
    def crossmerge1(
        self, 
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        B, C, H, W = x.shape
        L = H * W
        D, d_state = A_logs.shape
        K, D, dt_rank = dt_projs_weight.shape
        y = crossScan1.apply(x)
        y = crossMerge1.apply(y, H, W)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return y

    class crossScan2(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs
        
        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    
    class crossMerge2(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(ctx, ys: torch.Tensor, H, W):
            B, K, D, L = ys.shape
            ctx.shape = (H, W)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y
        
        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs, None, None

    def crossmerge2(
        self, 
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        B, C, H, W = x.shape
        L = H * W
        D, d_state = A_logs.shape
        K, D, dt_rank = dt_projs_weight.shape
        y = crossScan2.apply(x)
        y = crossMerge2.apply(y, H, W)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return y

    class CrossSelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx, 
            x: torch.Tensor = None, 
            x_proj_weight: torch.Tensor=None,
            x_proj_bias: torch.Tensor=None,
            dt_projs_weight: torch.Tensor=None,
            dt_projs_bias: torch.Tensor=None,
            A_logs: torch.Tensor=None,
            Ds: torch.Tensor=None,
            delta_softplus: bool = True,
            nrows = 1,
            H = 1, 
            W = 1,
        ):
            assert nrows in [1, 2, 4] # 8+ is too slow to compile
            B, C, L = x.shape
            K = 4
            KC = K * C
            N = A_logs.shape[-1]
            R = dt_projs_weight.shape[-1]
            ctx.shape = (B, C, H, W, L, K, N, R)
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows
            ctx.has_x_proj_bias = x_proj_bias is not None
            saved_tensors = []

            # ========== cross expand fwd: (B, C, H, W) -> (B, K, C, L)
            xs = x.new_empty((B, K, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

            # ========== linear fwd
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if ctx.has_x_proj_bias:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
            saved_tensors.extend([x_proj_weight, dt_projs_weight, x_dbl, xs])
 
            # ========== selective scan fwd
            u = xs.view(B, KC, L).to(torch.float)
            delta = dts.contiguous().view(B, KC, L).to(torch.float)
            As = -torch.exp(A_logs.to(torch.float))
            Bs = Bs.contiguous().to(torch.float)
            Cs = Cs.contiguous().to(torch.float)
            Ds = Ds.to(torch.float)
            delta_bias = dt_projs_bias.to(torch.float).view(-1)
            out, prefix, *rest = selective_scan_cuda.fwd(
                u,
                delta, 
                As, 
                Bs,
                Cs,
                Ds,
                delta_bias,
                delta_softplus,
                nrows,
            )
            ys = out.view(B, K, C, L)
            saved_tensors.extend([u, delta, As, Bs, Cs, Ds, delta_bias, prefix])
            
            # ========== cross merge fwd
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, C, L)
            y = ys[:, 0] + ys[:, 1].view(B, C, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, C, L)

            ctx.save_for_backward(*saved_tensors)
            return y

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, y: torch.Tensor):
            B, C, H, W, L, K, N, R = ctx.shape
            x_proj_weight, dt_projs_weight, fwdxdbl, fwdxs, fwdu, fwddelta, fwdA, fwdB, fwdC, fwdD, delta_bias, prefix = ctx.saved_tensors

            # ========== cross merge fwd
            xs = y.new_empty((B, 4, C, L))
            xs[:, 0] = y
            xs[:, 1] = y.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

            # ========== selective scan bwd
            xs = xs.contiguous().view(B, K*C, L)
            nrows = 1 # ctx.nrows
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                fwdu, fwddelta, fwdA, fwdB, fwdC, fwdD, delta_bias, xs, prefix, ctx.delta_softplus, nrows
            )

            # ========== linear bwd
            dx_dbl = du.new_empty((B, K, R + N + N, L))
            dx_dbl[:, :, :R, :] = torch.einsum("bkdl, kdr -> bkrl", ddelta.view(B, K, C, L), dt_projs_weight)
            dx_dbl[:, :, R:R + N, :] = dB
            dx_dbl[:, :, -N:, :] = dC
            dx = du.view(B, K, C, L)  + torch.einsum("bkcl,kcd -> bkdl", dx_dbl, x_proj_weight)
            ddt_projs_weight = torch.einsum("bkdl,bkrl->kdr", ddelta.view(B, K, C, L), fwdxdbl[:, :, :R, :])
            dx_proj_bias = torch.einsum("bkcl->kc", dx_dbl) if ctx.has_x_proj_bias else None
            dx_proj_weight = torch.einsum("bkcl,bkdl->kcd", dx_dbl, fwdxs)
            ddelta_bias = ddelta_bias.view(K, C)

            # ========== cross expand bwd: 
            dx = dx[:, 0:2] + dx[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            dx = dx[:, 0] + dx[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return dx, dx_proj_weight, dx_proj_bias, ddt_projs_weight, ddelta_bias, dA, dD, None, None, None, None

    def css(
        self,
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        B, C, H, W = x.shape
        y: torch.Tensor = CrossSelectiveScan.apply(
            x.flatten(2, 3),
            x_proj_weight, x_proj_bias, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, True, nrows, H, W,
        )
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y).to(x.dtype)
        return y

    def css1(
        self,
        x: torch.Tensor, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        nrows = 1,
    ):
        
        B, C, H, W = x.shape
        L = H * W
        D, d_state = A_logs.shape
        K, D, dt_rank = dt_projs_weight.shape

        xs = crossScan2.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        As = -torch.exp(A_logs.to(torch.float))  # (k * d, d_state)
        Ds = Ds.to(torch.float) # (k * d)
        dt_projs_bias = dt_projs_bias.to(torch.float).view(-1) # (k * d)
        ys: torch.Tensor = selective_scan(
                xs.to(torch.float), 
                dts.to(torch.float), 
                As, 
                Bs.to(torch.float), 
                Cs.to(torch.float), 
                Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                nrows=nrows,
            ).view(B, 4, -1, L)
        y = crossMerge2.apply(ys, H, W)
        y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = out_norm(y).to(x.dtype)
        
        return y


if True:
    import selective_scan_cuda_core as selective_scan_cuda
    
    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 4], f"{nrows}" # 8+ is too slow to compile
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)
            
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)

    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs
        
        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)
    
    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y
        
        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None

    def cross_selective_scan(
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        out_norm: torch.nn.Module=None,
        softmax_version=False,
        nrows = 1,
        delta_softplus = True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        xs = CrossScan.apply(x)
        
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)
         
        # to enable fvcore.nn.jit_analysis: inputs[i].debugName
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)
        
        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)
        
        return y


if True:
    def selective_scan_easy(us, dts, As, Bs, Cs, Ds, hprefix = None):
        """
        us: B, G * D, L
        dts: B, G * D, L
        As: G * D, N
        Bs: B, G, N, L
        Cs: B, G, N, L
        Ds: G * D
        """

        def selective_scan_chunk(us, dts, As, Bs, Cs, Ds, hprefix=None):
            """
            \partial(h) / \partial(t) = Ah + Bu; y = Ch + Du;
            => \partial(h*exp(-At)) / \partial(t) = Bu*exp(-At);
            => h_t = h_0*exp(At) + \sum_{t_0}_{t}_{Bu*exp(A(t-v)) dv};
            => h_b = exp(A(dt_a + ... + dt_{b-1})) * h_a
                   + exp(A(dt_a + ... + dt_{b-1})) * \sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i};
               y_i = C_i*h_i + D*u_i
            """
            """
            us: (L, B, G, D) # L is chunk_size
            dts: (L, B, G, D)
            As: (G, D, N)
            Bs: (L, B, G, N)
            Cs: (L, B, G, N)
            Ds: (G, D)
            hprefix: (1, B, G, D, N)
            """
            ts = dts.cumsum(dim=0)
            Ats = torch.einsum("gdn,lbgd->lbgdn", As, (ts[-2:-1] - ts)).exp()
            hs = torch.einsum("lbgn,lbgd,lbgdn,lbgd->lbgdn", Bs, us, Ats, dts)
            if hprefix is not None:
                hs = hs + torch.einsum("gdn,bgd->bgdn", As, ts[-1]).exp().unsqueeze(0) * hprefix
            hprefix = hs[-2:-1]
            ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs) + torch.einsum("gd,lbgd->lbgd", Ds, us)
            return ys, hprefix

        chunksize = 16
        B, G, N, L = Bs.shape
        us = us.view(B, G, -1, L).permute(3, 0, 1, 2)
        dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2)
        As = As.view(G, -1, N)
        Bs = Bs.permute(3, 0, 1, 2)
        Cs = Cs.permute(3, 0, 1, 2)
        Ds = Ds.view(G, -1)

        oys = []
        for i in range(0, L + chunksize - 1, chunksize):
            print("=====")
            ys, hprefix = selective_scan_chunk(
                us[i:i + chunksize], dts[i:i + chunksize], 
                As, Bs[i:i + chunksize], Cs[i:i + chunksize], Ds, hprefix, 
            )
            oys.append(ys)
        oys = torch.cat(oys, dim=0)
        return oys, hprefix



def test_selective_scan():
    
    device = torch.device("cuda")
    dtype = torch.float32
    B, H, W, D, N, R = 16, 56 // 4, 56 // 4, 768 * 2, 16, 192 // 16
    G, L = 4, H * W
    xi = torch.randn((B, D, H, W), device=device, dtype=dtype).requires_grad_()
    Ai = (-0.5 * torch.rand((G * D, N), device=device, dtype=dtype)).requires_grad_()
    Di = torch.randn((G * D), device=device, dtype=dtype).requires_grad_()
    xpw = torch.randn((G, (R + N + N), D), device=device, dtype=dtype).requires_grad_()
    xpb = torch.randn((G, (R + N + N)), device=device, dtype=dtype).requires_grad_()
    tpw = torch.randn((G, D, R), device=device, dtype=dtype).requires_grad_()
    tpb = torch.randn((G, D), device=device, dtype=dtype).requires_grad_()
    out_norm = torch.nn.LayerNorm(D, device=device, dtype=dtype)
    
    _xi = torch.randn((B, G * D, L), device=device, dtype=dtype).requires_grad_()
    dti = torch.randn((B, G * D, L), device=device, dtype=dtype).requires_grad_()
    Bi = torch.randn((B, G, N, L), device=device, dtype=dtype).requires_grad_()
    Ci = torch.randn((B, G, N, L), device=device, dtype=dtype).requires_grad_()
    _tpb = torch.randn((G * D), device=device, dtype=dtype).requires_grad_()

    def test_equal():
        xic = xi.detach().clone().requires_grad_()
        xpwc = xpw.detach().clone().requires_grad_()
        tpwc = tpw.detach().clone().requires_grad_()
        tpbc = tpb.detach().clone().requires_grad_()
        Aic = Ai.detach().clone().requires_grad_()
        Dic = Di.detach().clone().requires_grad_()
        import copy
        out_normc = copy.deepcopy(out_norm) 
        y1 = forward_corev1(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
        y2 = cross_selective_scan(xic, xpwc, None, tpwc, tpbc, Aic, Dic, out_normc)
        print((y1 - y2).abs().sum())
        y1.mean().backward()
        y2.mean().backward()
        
        for a, b in [(xi, xic), (xpw, xpwc), (tpw, tpwc), (tpb, tpbc), (Ai, Aic), (Di, Dic)]:
            print((a.grad - b.grad).abs().sum())

    test_equal()

    import time

    # nrow=1 or 4
    if False:
        tim0 = time.time()
        for _ in range(100):
            y = forward_corev1(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim1 = time.time()
        for _ in range(100):
            y = forward_corev1(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm, 4)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim2 = time.time()
        print(tim1-tim0, tim2-tim1, torch.cuda.max_memory_allocated())
        time.sleep(100000)
        # 3.531505584716797 3.4356584548950195 1402005504

    # cross scan and merge
    if False:
        tim0 = time.time()
        for _ in range(100):
            y = crossmerge(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim1 = time.time()
        for _ in range(100):
            y = crossmerge1(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim2 = time.time()
        for _ in range(100):
            y = crossmerge2(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim3 = time.time()
        print(tim1-tim0, tim2-tim1, tim3 - tim2, torch.cuda.max_memory_allocated())
        time.sleep(100000)
        # 0.496903657913208 0.2685220241546631 0.21291160583496094 583173120

    if False:
        tim0 = time.time()
        for _ in range(100):
            y = forward_corev1(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim1 = time.time()
        for _ in range(100):
            y = selective_scan(_xi, dti, Ai, Bi, Ci, Di, _tpb, True)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim2 = time.time()
        for _ in range(100):
            y = crossmerge(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim3 = time.time()
        for _ in range(100):
            y = linears(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim4 = time.time()
        for _ in range(100):
            y = css(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim5 = time.time()
        for _ in range(100):
            y = css1(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim6 = time.time()
        for _ in range(100):
            y = cross_selective_scan(xi, xpw, None, tpw, tpb, Ai, Di, out_norm, nrows=1)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim7 = time.time()
        for _ in range(100):
            y = cross_selective_scan(xi, xpw, None, tpw, tpb, Ai, Di, out_norm, nrows=4)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim8 = time.time()
        print(tim1-tim0, tim2-tim1, tim3 - tim2, tim4 - tim3, tim5 - tim4, tim6 - tim5, tim7 - tim6, tim8 - tim7, torch.cuda.max_memory_allocated())
        time.sleep(100000)
        # 3.5528743267059326 1.754795789718628 1.2213904857635498 0.7560632228851318 0.6416575908660889 0.5508723258972168 1402005504

    if True:
        tim0 = time.time()
        for _ in range(100):
            y = selective_scan(_xi, dti, Ai, Bi, Ci, Di, _tpb, True)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim1 = time.time()
        for _ in range(100):
            y = selective_scan_easy(_xi, dti, Ai, Bi, Ci, Di)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim2 = time.time()
        for _ in range(100):
            y = crossmerge(None, xi, xpw, None, tpw, tpb, Ai, Di, out_norm)
            y.mean().backward()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        tim3 = time.time()
        print(tim1-tim0, tim2-tim1, tim3 - tim2, torch.cuda.max_memory_allocated())
        time.sleep(100000)

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

# test_mamba_slow(16, 48, 100) # 360ms in 4090
# test_mamba_slow(64, 48, 100) # 648ms in 4090;
# test_mamba_slow(128, 48, 100) # 1063ms in 4090;

# test_cross_scan()
test_selective_scan()


