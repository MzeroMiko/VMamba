import torch

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
        return xs


import vssm_cross_scan_cuda

class CSv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        return vssm_cross_scan_cuda.cross_scan(x).view(B, 4, C, H * W)
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        ys = ys.view(B, 4, C, H, W)
        if ys.stride(-1) != 1:
            ys = ys.contiguous()
        return vssm_cross_scan_cuda.cross_merge(ys)


class CMv2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        return vssm_cross_scan_cuda.cross_merge(ys).view(B, D, H * W)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        if x.stride(-1) != 1:
            x = x.contiguous()
        return vssm_cross_scan_cuda.cross_scan(x.view(B, C, H, W))


def test():
    B, C, H, W = 128, 96, 56, 56
    x = torch.randn((B, C, H, W)).cuda().requires_grad_(True)
    y = torch.randn((B, 4, C, H, W)).cuda().requires_grad_(True)
    y1 = CrossScan.apply(x)
    y2 = CSv2.apply(x)
    x1 = CrossMerge.apply(y)
    x2 = CMv2.apply(y)
    print((y1 - y2).abs().sum())
    print((x1 - x2).abs().sum())
    print((x1 - x2).abs().max())

    for i in range(50):
        CrossScan.apply(x).sum().backward()
        CSv2.apply(x).sum().backward()
        CrossMerge.apply(y).sum().backward()
        CMv2.apply(y).sum().backward()
    print("warmup finish")
    import time
    t0 = time.time()
    torch.cuda.synchronize()
    for _ in range(1000):
        CrossScan.apply(x).sum().backward()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(1000):
        CSv2.apply(x).sum().backward()
    torch.cuda.synchronize()
    t2 = time.time()
    for _ in range(1000):
        CrossMerge.apply(y).sum().backward()
    torch.cuda.synchronize()
    t3 = time.time()
    for _ in range(1000):
        CMv2.apply(y).sum().backward()
    torch.cuda.synchronize()
    t4 = time.time()
    print(t4-t3,t3-t2,t2-t1,t1-t0)


test()

