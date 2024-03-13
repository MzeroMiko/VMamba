import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# triton cross scan, 2x speed than pytorch implementation =========================
try:
    from .csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1
except:
    from csm_triton import CrossScanTriton, CrossMergeTriton, CrossScanTriton1b1

# pytorch cross scan =============
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


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x
    
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb; pdb.set_trace()


# fvcore flops =======================================
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


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)

# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


# =============
# Note: we did not use csm_triton in and before vssm1_0230, we used pytorch version !
# Note: we did not use no_einsum in and before vssm1_0230, we used einsum version !
def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    channel_first=False,
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan,
    CrossMerge=CrossMerge,
    no_einsum=False, # replace einsum with linear or conv1d to raise throughput
    dt_low_rank=True,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    if (not dt_low_rank):
        x_dbl = F.conv1d(x.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, -1, L), [D, 4 * N, 4 * N], dim=1)
        xs = CrossScan.apply(x)
        dts = CrossScan.apply(dts)
    elif no_einsum:
        xs = CrossScan.apply(x)
        x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
    else:
        xs = CrossScan.apply(x)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous().view(B, K, N, L)
    Cs = Cs.contiguous().view(B, K, N, L)
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    
    y: torch.Tensor = CrossMerge.apply(ys)

    if channel_first:
        y = y.view(B, -1, H, W)
        if out_norm_shape in ["v1"]:
            y = out_norm(y)
        else:
            y = out_norm(y.permute(0, 2, 3, 1))
            y = y.permute(0, 3, 1, 2)
        return (y.to(x.dtype) if to_dtype else y)

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)


def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# =====================================================
# we have this class as linear and conv init differ from each other
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        # only used to run previous version
        if forward_type.startswith("v0"):
            self.__initv0__(d_model, d_state, ssm_ratio, dt_rank, dropout, seq=("seq" in forward_type))
            return
        
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = Linear2d if channel_first else nn.Linear

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        elif channel_first:
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
        )
        if forward_type.startswith("debug"):
            from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations, cross_selective_scanv2
            FORWARD_TYPES.update(dict(
                debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
                debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
                debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
                debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
                debugforward_core_mambassm_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
                debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
                debugforward_core_sscore_fusecscm_fwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fwdnrow, self),
                debugforward_core_sscore_fusecscm_bwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_bwdnrow, self),
                debugforward_core_sscore_fusecscm_fbnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fbnrow, self),
                debugforward_core_ssoflex_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
                debugforward_core_ssoflex_fusecscm_i16o32=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
                debugscan_sharessm=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scanv2),
            ))
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))
    
        if forward_type.startswith("xv"):
            self.d_state = d_state
            self.dt_rank = dt_rank
            self.d_inner = d_inner

            if d_conv > 1:
                self.conv2d = nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    groups=d_model,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                    **factory_kwargs,
                )
            self.act: nn.Module = act_layer()
            self.out_act: nn.Module = nn.Identity()
            del self.x_proj_weight

            # change Conv2d to Linear2d Next
            if forward_type.startswith("xv1"):
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = self.forwardxv

            if forward_type.startswith("xv2"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight

            if forward_type.startswith("xv3"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv4"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            if forward_type.startswith("xv5"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight
                self.out_act = nn.GELU()

            if forward_type.startswith("xv6"):
                self.forward = partial(self.forwardxv, mode="xv1")
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

    # only used to run previous version
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
            
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")

        return cross_selective_scan(
            x, x_proj_weight, None, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, delta_softplus=True,
            out_norm=out_norm,
            channel_first=self.channel_first,
            out_norm_shape=out_norm_shape,
            **kwargs,
        )
    
    # only used to run previous version
    def forwardv0(self, x: torch.Tensor, SelectiveScan = SelectiveScanMamba, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        if not self.channel_first:
            y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
            y = self.out_norm(y).view(B, H, W, -1)
        else:
            y = self.out_norm(y.view(B, -1, H, W))

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out
    
    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        y = self.forward_core(x)

        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def forwardxv(self, x: torch.Tensor, mode="xv1", **kwargs):
        B, C, H, W = x.shape
        if not self.channel_first:
            B, H, W, C = x.shape
        L = H * W
        K = 4
        dt_projs_weight = getattr(self, "dt_projs_weight", None)
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x = self.conv2d(x) # (b, d, h, w)
            x = self.act(x)
        x = self.in_proj(x)

        if mode in ["xv1"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, -1, L)
            dts = F.conv1d(dts, dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
            # below is slower
            # us_dts, Bs, Cs = x.split([self.d_inner + self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            # us_dts = CrossScanTriton.apply(us_dts.contiguous())
            # us = us_dts[:, :, :self.d_inner, :].contiguous().view(B, -1, L)
            # dts = us_dts[:, :, self.d_inner:, :].contiguous().view(B, -1, L)
            # dts = F.conv1d(dts, dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        elif mode in ["xv2"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts).contiguous().view(B, -1, L)
        elif mode in ["xv3"]:
            us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton1b1.apply(dts.contiguous().view(B, K, -1, H, W))
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        else:
            ...

        Bs, Cs = Bs.view(B, K, -1, L).contiguous(), Cs.view(B, K, -1, L).contiguous()
    
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
            
        y: torch.Tensor = CrossMergeTriton.apply(ys)
        y = y.view(B, -1, H, W)

        # originally:
        # y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        # y = out_norm(y).view(B, H, W, -1)

        if (not self.channel_first) or (out_norm_shape in ["v0"]):
            y = out_norm(y.permute(0, 2, 3, 1))
            if self.channel_first:
                y = y.permute(0, 3, 1, 2)
        else:
            y = out_norm(y)

        y = (y.to(x.dtype) if to_dtype else y)
        out = self.dropout(self.out_proj(self.out_act(y)))
        return out


if False:
# if True:
    try:
        from .ss2d_ablations import SS2DDev
        SS2D = SS2DDev
        print("DEBUG MODE ===========================================")
    except Exception as e:
        if isinstance(e, ImportError):
            pass
        print(e, flush=True)
        pass


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(self.drop(x) * self.act(z))
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN", # "BN", "LN2D"
        downsample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer, channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D, 
            v2=self._make_downsample, 
            v3=self._make_downsample_v3, 
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        seq = [nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),]
        if not channel_first:
            seq.append(Permute(0, 2, 3, 1))
        if patch_norm:
            seq.append(norm_layer(embed_dim))
        return nn.Sequential(*seq)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        assert patch_size == 4
        seq = [nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1)]
        if patch_norm:
            seq.extend([
                Permute(0, 2, 3, 1), 
                norm_layer(embed_dim // 2), 
                Permute(0, 3, 1, 2)
            ] if not channel_first else [norm_layer(embed_dim // 2)])
        seq.extend([
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
        ])
        if not channel_first:
            seq.append(Permute(0, 2, 3, 1))
        if patch_norm:
            seq.append(norm_layer(embed_dim))
        return nn.Sequential(*seq)
    
    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        ) if not channel_first else nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            Permute(0, 3, 1, 2),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            norm_layer(out_dim),
        ) if not channel_first else nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer=nn.LayerNorm, **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        
        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)        
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        if len(self.out_indices) == 0:
            return x
        
        return outs


# ==================================================
class CHECKS:
    def check_vssm_equals_vmambadp():
        try:
            from _ignore.vmamba.vmamba_bak1 import VMamba2Dp
            from _ignore.vmamba.vmamba_pub import VSSM
        except:
            print("original VSSM and VMamba2Dp not found.", flush=True)
            return 

        # test 1 True =================================
        torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
        oldvss = VMamba2Dp(depths=[2,2,6,2]).half().cuda()
        newvss = VSSM(depths=[2,2,6,2]).half().cuda()
        newvss.load_state_dict(oldvss.state_dict())
        input = torch.randn((12, 3, 224, 224)).half().cuda()
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward_backbone(input)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward_backbone(input)
        print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward(input)
        torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward(input)
        print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        
        # test 2 True ==========================================
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        oldvss = VMamba2Dp(depths=[2,2,6,2]).cuda()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        newvss = VSSM(depths=[2,2,6,2]).cuda()

        miss_align = 0
        for k, v in oldvss.state_dict().items(): 
            same = (oldvss.state_dict()[k] == newvss.state_dict()[k]).all()
            if not same:
                print(k, same)
                miss_align += 1
        print("init miss align", miss_align) # init miss align 0

    def check_vssm1_equals_vssm(forward_type="v0"):
        try:
            from _ignore.vmamba.vmamba_pub import VSSM as VSSM0
        except:
            print("original VSSM and VMamba2Dp not found.", flush=True)
            return

        class VSSM_(VSSM):
            @staticmethod
            def _make_layer(*args, **kwargs):
                layer = VSSM._make_layer(*args, **kwargs)
                dim = kwargs.get("dim", None)
                norm_layer = kwargs.get("norm_layer", None)
                downsample = kwargs.get("downsample", None)
                blocks = layer.blocks
            
                if True: # is this really applied? Yes, but been overriden later in VSSM!
                    def _init_weights(module: nn.Module):
                        for name, p in module.named_parameters():
                            if name in ["out_proj.weight"]:
                                p = p.clone().detach_() # fake init, just to keep the seed ....
                                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    blks = nn.Sequential(*copy.deepcopy(blocks))
                    blks.apply(_init_weights)

                downsample = PatchMerging2D(dim, 2*dim, norm_layer=norm_layer) if downsample is None else nn.Identity()
                
                return nn.Sequential(OrderedDict(
                    blocks=nn.Sequential(*blocks,),
                    downsample=downsample,
                ))

            def forward_backbone(self, x):
                x = self.patch_embed(x)
                for l in self.layers:
                    x = l(x)
                return x

            def forward1(self, x: torch.Tensor):
                x = self.patch_embed(x)
                for layer in self.layers:
                    x = layer(x)
                x = self.classifier.norm(x)
                # here: whether has contiguous would differ
                x = self.classifier.avgpool(x.permute(0, 3, 1, 2).contiguous()).flatten(1)
                x = self.classifier.head(x)
                return x

        # only has initial difference 
        VSSM1 = partial(VSSM, downsample_version="v1", patchembed_version="v1", mlp_ratio=0.0, ssm_ratio=2.0, forward_type=forward_type)
        VSSM.forward_backbone = VSSM_.forward_backbone 
        VSSM.forward1 = VSSM_.forward1
        # expected to be all the same 
        VSSM1 = partial(VSSM_, downsample_version="none", patchembed_version="v1", mlp_ratio=0.0, ssm_ratio=2.0, forward_type=forward_type)

        # test 1 True =================================
        torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
        oldvss = VSSM0(depths=[2,2,6,2]).half().cuda()
        newvss = VSSM1(depths=[2,2,6,2]).half().cuda()
        newvss.load_state_dict(oldvss.state_dict())
        input = torch.randn((12, 3, 224, 224)).half().cuda()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward_backbone(input)
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward_backbone(input)
        print((y1 -y2).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward(input)
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward1(input)
        print((y1 -y2).abs().sum()) # tensor(2.5988e-05, device='cuda:0', grad_fn=<SumBackward0>)
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y3 = newvss.forward(input)
        print((y1 -y3).abs().sum()) # tensor(0., device='cuda:0', grad_fn=<SumBackward0>)
        
        # test 2 True ==========================================
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        oldvss = VSSM0(depths=[2,2,6,2]).cuda()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        newvss = VSSM1(depths=[2,2,6,2]).cuda()

        miss_align = 0
        oldvss2new = copy.deepcopy(newvss)
        oldvss2new.load_state_dict(oldvss.state_dict())
        for k, v in oldvss2new.state_dict().items(): 
            same = (oldvss2new.state_dict()[k] == newvss.state_dict()[k]).all()
            if not same:
                print(k, same)
                miss_align += 1
        print("init miss align", miss_align) # init miss align 0

    def check_vssm1_ssoflex_equals_mambassm():
        # only has initial difference
        VSSM0 = partial(VSSM, downsample_version="v3", patchembed_version="v2", mlp_ratio=4.0, ssm_ratio=2.0, forward_type="v2")
        VSSM1 = partial(VSSM, downsample_version="v3", patchembed_version="v2", mlp_ratio=4.0, ssm_ratio=2.0, forward_type="v01")

        # test 1 True =================================
        torch.manual_seed(time.time()); torch.cuda.manual_seed(time.time())
        oldvss = VSSM0(depths=[2,2,6,2]).half().cuda()
        newvss = VSSM1(depths=[2,2,6,2]).half().cuda()
        newvss.load_state_dict(oldvss.state_dict())
        input0 = torch.randn((12, 3, 224, 224)).half().cuda().requires_grad_()
        input1 = input0.detach().clone().requires_grad_()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y1 = oldvss.forward(input0)
            y1.sum().backward()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        with torch.cuda.amp.autocast():
            y2 = newvss.forward(input1)
            y2.sum().backward()
        print((y1 - y2).abs().sum()) # tensor(0., device='cuda:0', dtype=torch.float16, grad_fn=<SumBackward0>)
        print((input0.grad - input1.grad).abs().sum()) # tensor(6.6016, device='cuda:0', dtype=torch.float16)
        
        # test 2 True ==========================================
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        oldvss = VSSM0(depths=[2,2,6,2]).cuda()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        newvss = VSSM1(depths=[2,2,6,2]).cuda()

        miss_align = 0
        oldvss2new = copy.deepcopy(newvss)
        oldvss2new.load_state_dict(oldvss.state_dict())
        for k, v in oldvss2new.state_dict().items(): 
            same = (oldvss2new.state_dict()[k] == newvss.state_dict()[k]).all()
            if not same:
                print(k, same)
                miss_align += 1
        print("init miss align", miss_align) # init miss align 0

    def check_csm_triton():

        B, C, H, W = 128, 192, 56, 57
        dtype=torch.float16
        dtype=torch.float32
        x = torch.randn((B, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        y = torch.randn((B, 4, C, H, W), dtype=dtype, device=torch.device("cuda")).requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        y1 = y.clone().detach().requires_grad_(True)

        def cross_scan(x: torch.Tensor):
            B, C, H, W = x.shape
            L = H * W
            xs = torch.stack([
                x.view(B, C, L),
                torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L),
                torch.flip(x.contiguous().view(B, C, L), dims=[-1]),
                torch.flip(torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
            ], dim=1).view(B, 4, C, L)
            return xs
        
        def cross_merge(out_y: torch.Tensor):
            B, K, D, H, W = out_y.shape
            L = H * W
            out_y = out_y.view(B, K, D, L)
            inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
            wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
            y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
            return y

        if True:
            res0 = triton.testing.do_bench(lambda :cross_scan(x))
            res1 = triton.testing.do_bench(lambda :CrossScan.apply(x))
            res2 = triton.testing.do_bench(lambda :CrossScanTriton.apply(x))
            res3 = triton.testing.do_bench(lambda :cross_merge(y))
            res4 = triton.testing.do_bench(lambda :CrossMerge.apply(y))
            res5 = triton.testing.do_bench(lambda :CrossMergeTriton.apply(y))
            print(res0, res1, res2, res3, res4, res5)

        print("test cross scan")
        if True:
            o0 = cross_scan(x)
            o1 = CrossScanTriton.apply(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            o0 = cross_merge(y)
            o1 = CrossMergeTriton.apply(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None

        print("test cross scan 1d")
        if False:
            def cross_scan_1d(x: torch.Tensor):
                B, C, H, W = x.shape
                return torch.stack([x,x,x,x], dim=1).view(B, 4, C, -1)
            def cross_merge_1d(x: torch.Tensor):
                return x.sum(dim=1).flatten(2, 3)
            o0 = cross_scan_1d(x)
            o1 = CrossScanTriton.apply(x1, 5)
            o1 = CrossScan_Ab_1direction.apply(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cross_merge_1d(y)
            o1 = CrossMergeTriton.apply(y1, 5)
            o1 = CrossMerge_Ab_1direction.apply(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max()) # failed with CrossMergeTriton
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
        
        print("test cross scan 2d")
        if False:
            def cross_scan_2d(x: torch.Tensor):
                B, C, H, W = x.shape
                x = x.flatten(2, 3)
                return torch.stack([x,x,x.flip(dims=[-1]), x.flip(dims=[-1])], dim=1).view(B, 4, C, -1)
            def cross_merge_2d(out_y: torch.Tensor):
                B, K, D, H, W = out_y.shape
                L = H * W
                out_y = out_y.view(B, K, D, L)
                inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
                y = (out_y[:, 0:2] + inv_y[:, 0:2]).sum(1)
                return y
            
            o0 = cross_scan_2d(x)
            o1 = CrossScanTriton.apply(x1, 7)
            o1 = CrossScan_Ab_2direction.apply(x1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((x.grad - x1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
            o0 = cross_merge_2d(y)
            o1 = CrossMergeTriton.apply(y1, 7)
            o1 = CrossMerge_Ab_2direction.apply(y1)
            o0.backward(x.view(B, C, H * W))
            o1.backward(x.view(B, C, H * W))
            print((o0 - o1).abs().max()) # failed with CrossMergeTriton
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None
        
        print("test cross scan one by one")
        if True:
            def cross_scan_1b1(x: torch.Tensor):
                B, K, C, H, W = x.shape
                L = H * W
                xs = torch.stack([
                    x[:, 0].view(B, C, L),
                    torch.transpose(x[:, 1], dim0=2, dim1=3).contiguous().view(B, C, L),
                    torch.flip(x[:, 2].contiguous().view(B, C, L), dims=[-1]),
                    torch.flip(torch.transpose(x[:, 3], dim0=2, dim1=3).contiguous().view(B, C, L), dims=[-1]),
                ], dim=1).view(B, 4, C, L)
                return xs
            o0 = cross_scan_1b1(y)
            o1 = CrossScanTriton1b1.apply(y1)
            o0.backward(y.view(B, 4, C, H * W))
            o1.backward(y.view(B, 4, C, H * W))
            print((o0 - o1).abs().max())
            print((y.grad - y1.grad).abs().max())
            x.grad, x1.grad, y.grad, y1.grad = None, None, None, None

    def check_einsum():
        B, D, L, R, K = 128, 192, 56 * 56, 12, 4
        o = torch.randn((B, K * D, L)).cuda()
        x = torch.randn((B, K, R, L)).cuda().requires_grad_(True)
        w = torch.randn((K, D, R)).cuda().requires_grad_(True)
        x1 = x.clone().detach().requires_grad_(True)
        w1 = w.clone().detach().requires_grad_(True)

        y1 = torch.einsum("bkrl,kdr->bkdl", x, w).contiguous().view(B, -1, L)
        y2 = F.conv1d(x1.view(B, -1, L), w1.view(K * D, R, 1), None, groups=K).contiguous().view(B, -1, L)
        print((y1 - y2).abs().max())
        y1.backward(o)
        y2.backward(o)
        print((x.grad - x1.grad).abs().max())

    def check_vssblock():
        import triton
        from torchvision.models.vision_transformer import EncoderBlock

        vb = VSSBlock(
            hidden_dim=16, 
            drop_path=0.0, 
            norm_layer=nn.LayerNorm, 
            ssm_d_state=1, 
            ssm_ratio=2, 
            ssm_dt_rank="auto", 
            ssm_act_layer=nn.SiLU,
            ssm_conv=3, 
            ssm_conv_bias=False, 
            ssm_drop_rate=0.0, 
            ssm_init="v0", 
            forward_type="v2", 
            mlp_ratio=4, 
            mlp_act_layer=nn.GELU, 
            mlp_drop_rate=0.0, 
            use_checkpoint=False,
        ).cuda()
        
        trans = EncoderBlock(
            num_heads=1, 
            hidden_dim=16, 
            mlp_dim=int(4.0 * 16), 
            dropout=0.0, 
            attention_dropout=0.0, 
            norm_layer=nn.LayerNorm,
        ).cuda()

        inp = torch.randn((16, 128, 128, 16)).cuda().requires_grad_()
        inp2 = inp.detach().cuda().view(16, -1, 16).requires_grad_()
        fn = lambda :vb(inp)
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        fn = lambda :trans(inp2)
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        fn = lambda :vb(inp).sum().backward()
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        fn = lambda :trans(inp2).sum().backward()
        ms = triton.testing.do_bench(fn, warmup=100)
        print(ms)
        import time; time.sleep(10000)

    def check_ln2d():
        import triton
        inp = torch.randn((64, 192, 56, 57)).cuda().requires_grad_()
        inp2 = inp.detach().permute(0, 2, 3, 1).clone().requires_grad_()

        torch.manual_seed(0); torch.cuda.manual_seed(0)
        n1 = LayerNorm2d(192).cuda()
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        n2 = nn.LayerNorm(192).cuda()
        o1 = n1(inp)
        o2 = n2(inp2)
        print((o1.permute(0, 2, 3, 1) - o2).abs().max())
        o1.backward(inp.data)
        o2.backward(inp.data.permute(0, 2, 3, 1))
        print((inp.grad.permute(0, 2, 3, 1) - inp2.grad).abs().max())

        ms1 = triton.testing.do_bench(lambda:n1(inp))
        ms2 = triton.testing.do_bench(lambda:n2(inp2))
        ms3 = triton.testing.do_bench(lambda:n1(inp))
        print(ms1, ms2, ms3)

    def check_linear_2d():
        import triton
        inp = torch.randn((64, 192, 56, 57)).cuda().requires_grad_()
        inp2 = inp.detach().permute(0, 2, 3, 1).clone().requires_grad_()

        torch.manual_seed(0); torch.cuda.manual_seed(0)
        n1 = Mlp(192, 4*192, 384, channels_first=True).cuda()
        catch_random1 = torch.randn((1,))
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        n2 = Mlp(192, 4*192, 384, channels_first=False).cuda()
        catch_random2 = torch.randn((1,))
        print(catch_random1, catch_random2)
        with torch.cuda.amp.autocast():
            o1 = n1(inp)
            o2 = n2(inp2)
        print((o1.permute(0, 2, 3, 1) - o2).abs().max())
        o1.sum().backward()
        o2.sum().backward()
        print((inp.grad.permute(0, 2, 3, 1) - inp2.grad).abs().max())

        i1, i2 = inp.float(), inp2.float()
        ms2 = triton.testing.do_bench(lambda:n2(i2))
        ms1 = triton.testing.do_bench(lambda:n1(i1))
        ms4 = triton.testing.do_bench(lambda:n2(i2).sum().backward())
        ms3 = triton.testing.do_bench(lambda:n1(i1).sum().backward())
        print(ms1, ms2, ms3, ms4)

    def check_channel_first():
        import triton
        inp = torch.randn((64, 3, 224, 224)).cuda().half().requires_grad_()
        inp2 = inp.detach().clone().requires_grad_()

        torch.manual_seed(0); torch.cuda.manual_seed(0)
        n1 = VSSM(norm_layer="ln").cuda()
        catch_random1 = torch.randn((1,))
        torch.manual_seed(0); torch.cuda.manual_seed(0)
        n2 = VSSM(norm_layer="ln2d").cuda()
        catch_random2 = torch.randn((1,))
        print(catch_random1, catch_random2)
        with torch.cuda.amp.autocast():
            # o1 = nn.Sequential(*n1.layers)(n1.patch_embed(inp))
            # o2 = nn.Sequential(*n2.layers)(n2.patch_embed(inp2))
            # o1 = n1.layers[0](n1.patch_embed(inp))
            # o2 = n2.layers[0](n2.patch_embed(inp2))
            # o1 = n1.layers[2](n1.layers[1](n1.layers[0](n1.patch_embed(inp))))
            # o2 = n2.layers[2](n2.layers[1](n2.layers[0](n2.patch_embed(inp2))))
            _n1 = lambda x:n1.layers[3].blocks[0].norm(n1.layers[2](n1.layers[1](n1.layers[0](n1.patch_embed(x)))))
            _n2 = lambda x:n2.layers[3].blocks[0].norm(n2.layers[2](n2.layers[1](n2.layers[0](n2.patch_embed(x)))))
            o1 = n1.layers[3].blocks[0].op(_n1(inp))
            o2 = n2.layers[3].blocks[0].op(_n2(inp2))
            o1 = _n1(inp)
            o2 = _n2(inp2)
        print((o1.abs().sum() - o2.abs().sum()).abs().max())
        o1.sum().backward()
        o2.sum().backward()
        print((inp.grad - inp2.grad).abs().max())
        breakpoint()

        i1, i2 = inp.float(), inp2.float()
        ms2 = triton.testing.do_bench(lambda:n2(i2))
        ms1 = triton.testing.do_bench(lambda:n1(i1))
        ms4 = triton.testing.do_bench(lambda:n2(i2).sum().backward())
        ms3 = triton.testing.do_bench(lambda:n1(i1).sum().backward())
        print(ms1, ms2, ms3, ms4)

    def check_profile():
        vss = VSSM(depths=[1], dims=1024).half().cuda()
        input = torch.randn((128, 3, 56, 56)).half().cuda()
        torch.cuda.manual_seed(0)

        def trace_handler(prof: torch.profiler.profile):
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
            # print(prof.export_chrome_trace("./tracev1.json"))

        with torch.cuda.amp.autocast():
            # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=True, with_stack=True) as prof:
            with torch.profiler.profile(
                with_modules=True,
                with_stack=True,
                profile_memory=True,
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],

                # In this example with wait=1, warmup=1, active=2, repeat=1,
                # profiler will skip the first step/iteration,
                # start warming up on the second, record
                # the third and the forth iterations,
                # after which the trace will become available
                # and on_trace_ready (when set) is called;
                # the cycle repeats starting with the next step

                schedule=torch.profiler.schedule(
                    wait=1,
                    warmup=1,
                    active=2,
                    repeat=1),
                on_trace_ready=trace_handler
                # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
                # used when outputting for tensorboard
                ) as prof:
                    for iter in range(1000):
                        x = input
                        # with torch.autograd.profiler.record_function("patch_embed"):
                        #     x = self.patch_embed(x)
                        prof.step()

    def load22kto1k():
        if False:
            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete relative_coords_table since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete attn_mask since we always re-init it
            attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del state_dict[k]

            # bicubic interpolate relative_position_bias_table if not match
            relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        # bicubic interpolate relative_position_bias_table if not match
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

            # bicubic interpolate absolute_pos_embed if not match
            absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
            for k in absolute_pos_embed_keys:
                # dpe
                absolute_pos_embed_pretrained = state_dict[k]
                absolute_pos_embed_current = model.state_dict()[k]
                _, L1, C1 = absolute_pos_embed_pretrained.size()
                _, L2, C2 = absolute_pos_embed_current.size()
                if C1 != C1:
                    logger.warning(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                        absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                            absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                        state_dict[k] = absolute_pos_embed_pretrained_resized

            # check classifier, if not match, then re-init classifier to zero
            head_bias_pretrained = state_dict['head.bias']
            Nc1 = head_bias_pretrained.shape[0]
            Nc2 = model.head.bias.shape[0]
            if (Nc1 != Nc2):
                if Nc1 == 21841 and Nc2 == 1000:
                    logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
                    map22kto1k_path = f'data/map22kto1k.txt'
                    with open(map22kto1k_path) as f:
                        map22kto1k = f.readlines()
                    map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
                    state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
                    state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
                else:
                    torch.nn.init.constant_(model.head.bias, 0.)
                    torch.nn.init.constant_(model.head.weight, 0.)
                    del state_dict['head.weight']
                    del state_dict['head.bias']
                    logger.warning(f"Error in loading classifier head, re-init classifier head to 0")


if __name__ == "__main__":
    import triton

    # CHECKS.check_vssblock()
    # CHECKS.check_vssm_equals_vmambadp()
    # CHECKS.check_vssm1_equals_vssm(forward_type="v0")
    # CHECKS.check_vssm1_equals_vssm(forward_type="v0_seq")
    # CHECKS.check_vssm1_equals_vssm(forward_type="v2")
    # print(VSSM(forward_type="v0").flops())
    # print(VSSM(forward_type="v2").flops())
    # print(VSSM(forward_type="v2nozact").flops())

    # CHECKS.check_vssm1_ssoflex_equals_mambassm()
    # CHECKS.check_csm_triton()
    # CHECKS.check_einsum()
    CHECKS.check_ln2d()
    CHECKS.check_linear_2d()
    CHECKS.check_channel_first()


    
    # breakpoint()

    


