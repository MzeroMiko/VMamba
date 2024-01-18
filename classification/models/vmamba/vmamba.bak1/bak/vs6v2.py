import math
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.vision_transformer import VisionTransformer, MLPBlock, EncoderBlock
from functools import partial
from einops import rearrange, repeat

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat
import torch.cuda as cuda

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except ImportError:
    selective_scan_fn, mamba_inner_fn = None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref, mamba_inner_fn, mamba_inner_ref
from mamba_ssm.models.mixer_seq_simple import _init_weights, Mamba, create_block

try:
    from .vs6 import flops_selective_scan_ref, flops_mamba, MambaBlock
except:
    from vs6 import flops_selective_scan_ref, flops_mamba, MambaBlock


# ========================================

class MambaRef0(Mamba):
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if True:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if True:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out


class MambaRef1(Mamba):
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if True:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if True:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_ref(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out


class MambaBlockRef0(nn.Module):
    ATTNBLOCK = MambaRef0
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int = 0,
        hidden_dim: int = 0,
        mlp_dim: int = 0,
        dropout: float = 0,
        attention_dropout: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim,use_fast_path=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input
        return x


class MambaBlockRef1(MambaBlockRef0):
    ATTNBLOCK = MambaRef1


def selective_scan_ref2(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    
    print("sleep")
    # 15G
    deltaAs = torch.cumprod(deltaA[:, :, :], dim=-1)
    # 24G
    import time; time.sleep(100000)
    
    
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


def batch_selective_scan_fn(
    u, delta, A, B, C,
    D=None, z=None, delta_bias=None, 
    delta_softplus=True, return_last_state=False 
):
    # shape assertions ====================
    k, b, d, l = u.shape
    d_state = A.shape[-1]

    assert (k, b, d, l) == delta.shape
    assert (k, b, d_state, l) == B.shape
    assert (k, b, d_state, l) == C.shape
    assert (k, d) == A.shape[:2]

    if D is None:
        D = [None] * k
    else:
        assert (k, d) == D.shape

    if z is None:
        z = [None] * k
    else:
        assert(k, b, d, l) == z.shape

    if delta_bias is None:
        delta_bias = [None] * k
    else:
        assert (k, d) == delta_bias.shape


    # baseline version
    if False:
        outputs = torch.zeros(k, b, d, l, device=u.device, dtype=u.dtype)
        for i in range(k):
            outputs[i] = selective_scan_fn(
                    u[i], delta[i], A[i], B[i], C[i], D[i], z[i], delta_bias[i],
                    delta_softplus=delta_softplus, return_last_state=return_last_state,
                )

    # multi_stream version, but slower
    if True:
        num_streams = k
        streams = [cuda.Stream() for _ in range(num_streams)]

        outputs = torch.zeros(k, b, d, l, device=u.device, dtype=u.dtype)

        for i in range(num_streams):
            with cuda.stream(streams[i]):
                outputs[i] = selective_scan_fn(
                    u[i], delta[i], A[i], B[i], C[i], D[i], z[i], delta_bias[i],
                    delta_softplus=delta_softplus, return_last_state=return_last_state,
                )

        for i in range(num_streams):
            streams[i].synchronize()

    # numba_version
            
    import numba
    from numba import cuda
    if False:
        ...


    return outputs


class MambaRef2(Mamba):
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if True:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if True:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_ref2(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out


class MambaBlockRef2(MambaBlockRef0):
    ATTNBLOCK = MambaRef2


class BiMambaFast(Mamba):
    def forward(self, x, **kwargs):
        B, L, D = x.shape
        invx = torch.flip(x, dims=[1])
        x = torch.cat([x, invx], dim=0)
        out = super().forward(x)
        out = out.view(2, B, L, D).sum(dim=0)
        return out


class BiMambaBlockFast(MambaBlock):
    ATTNBLOCK = BiMambaFast
    def __init__(
        self,
        num_heads: int = 0,
        hidden_dim: int = 0,
        mlp_dim: int = 0,
        dropout: float = 0,
        attention_dropout: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim, use_fast_path=True)
        self.dropout = nn.Dropout(dropout)


class BiMamba2DFast(nn.Module):
    def __init__(self, d_model, d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
    ):
        super().__init__()
        kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
        )
        self.h_bimamba = BiMambaFast(**kwargs)
        self.w_bimamba = BiMambaFast(**kwargs)

    def forward(self, x, **kwargs):
        B, H, W, D = x.shape
        xw = torch.transpose(x, dim0=1, dim1=2).contiguous().view(-1, H, D)
        xw = self.h_bimamba(xw)
        xh = torch.transpose(xw, dim0=1, dim1=2).contiguous().view(-1, H, D)
        xh = self.w_bimamba(xh)
        y = xh.view(B, H, W, D)
        return y


class BiMamba2DFast1(nn.Module):
    def __init__(self, d_model, d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
    ):
        super().__init__()
        kwargs = dict(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
        )
        self.w_bimamba = BiMambaFast(**kwargs)

    def forward(self, x, **kwargs):
        B, H, W, D = x.shape
        xhw = x.view(B, H * H, D)
        xhwinv = torch.flip(xhw, dims=[1])
        xwh = torch.transpose(x, dim0=1, dim1=2).contiguous().view(B, W * H, D)
        xwhinv = torch.flip(xwh, dims=[1])
        x = torch.stack([xhw, xwh, xhwinv, xwhinv], dim=0)
        y = self.w_bimamba(x.view(4 * B, H * W, D))
        y = y.view(4, B, H * W, D)
        y_hw_wh = y[0:2] + torch.flip(y[2:4], dims=[2])
        y = y_hw_wh[0].view(B, H, W, D) + torch.transpose(y_hw_wh[1].view(B, W, H, D), dim0=1, dim1=2).contiguous()
        return y


def selective_scan_base(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = True
    is_variable_C = True
    B = B.float()
    C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    if False:
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
    
    if True:
        oxs = []
        # for i in range(u.shape[2]):
            # x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            # oxs.append(x)
        # oxs = torch.stack(oxs, dim=-1) 
        oxs = deltaB_u.permute(0,1,3,2) + deltaA.permute(0,1,3,2)
        y = torch.einsum('bdnl,bnl->bdl', oxs, C[:, :, :])
    
    return y

# expand = 1:
# INFO number of params: 10822024
# INFO number of GFLOPs: 1.882518528
# expand = 2:
# INFO number of params: 19306120
# INFO number of GFLOPs: 3.574001664
class BiMamba2DSlow4(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4) # (K=4, D, N)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs):
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
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        x_hw = x
        x_hwwh = torch.stack([x_hw, torch.transpose(x_hw, dim0=2, dim1=3).contiguous()], dim=0).view(2, B, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[3])], dim=0) # (k, b, d, l)

        As = -torch.exp(self.A_logs.float())  # (4, inner, state)
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs_bias.float()

        x_dbl = torch.einsum("k b d l, k c d -> k b c l", xs.view(4, B, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(4, 1, -1, 1) # (k, b, d, l)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("k b r l, k d r -> k b d l", dts.view(4, B, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(4, 1, -1, 1)

        # added 20231222
        if True:
            xs = xs.float()
            dts = dts.float()
            Bs = Bs.float()
            Cs = Cs.float()
            z = z.float()

        hw_y = selective_scan_fn(
                xs[0], dts[0], 
                As[0], Bs[0], Cs[0], Ds[0], z=None,
                delta_bias=dt_projs_bias[0],
                delta_softplus=True,
                return_last_state=False,
            )
        wh_y = selective_scan_fn(
                xs[1], dts[1], 
                As[1], Bs[1], Cs[1], Ds[1], z=None,
                delta_bias=dt_projs_bias[1],
                delta_softplus=True,
                return_last_state=False,
            )
        invhw_y = selective_scan_fn(
                xs[2], dts[2], 
                As[2], Bs[2], Cs[2], Ds[2], z=None,
                delta_bias=dt_projs_bias[2],
                delta_softplus=True,
                return_last_state=False,
            )
        invwh_y = selective_scan_fn(
                xs[3], dts[3], 
                As[3], Bs[3], Cs[3], Ds[3], z=None,
                delta_bias=dt_projs_bias[3],
                delta_softplus=True,
                return_last_state=False,
            )

        wh_y = torch.transpose(wh_y.view(B, -1, W, H), dim0=2, dim1=3).contiguous()
        invwh_y = torch.transpose(torch.flip(invwh_y, dims=[2]).view(B, -1, W, H), dim0=2, dim1=3).contiguous()
        invhw_y = torch.flip(invhw_y, dims=[2])

        # added 20231222
        if True:
            y = hw_y.float() + invhw_y.float() + wh_y.float().view(B, -1, L) + invwh_y.float().view(B, -1, L)
            y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
            y = self.out_norm(y)
        else:
            y = hw_y + invhw_y + wh_y.view(B, -1, L) + invwh_y.view(B, -1, L) # ALL IN FP16
            y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
            y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out

    def flops(self, B=1, H=16, W=16):
        L = H * W
        d_model = self.d_model
        d_inner = self.d_inner
        d_conv = self.d_conv
        dt_rank = self.dt_rank
        d_state = self.d_state

        flops = 0
        flops += B * L * d_model * (d_inner * 2) # xz = ...
        cg = d_inner # conv group
        cgci, cgco = d_inner // cg, d_inner // cg # conv_group_channel_in, conv_group_channel_out
        flops += B * L * (cg * (cgci * cgco * d_conv * d_conv)) # x = self.act(self.conv1d(x)[..., :seqlen])
        flops += 4 * B * L * d_inner * (dt_rank + 2 * d_state) # x_dbl = ...
        flops += 4 * B * L * dt_rank * d_inner # dt =
        flops += 4 * flops_selective_scan_ref(B=B, L=L, D=d_inner, N=d_state, with_D=True, with_Z=False) # y = selective_scan_fn( 
        flops += B * L * d_inner # y = self.out_norm(y)
        flops += B * L * d_inner * d_model    # out = self.out_proj(y)
        # print(flops, flops - 4 * flops_selective_scan_ref(B=B, L=L, D=d_inner, N=d_state, with_D=True, with_Z=False))
        return flops

    def forward_multi_stream(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        x_hw = x
        x_hwwh = torch.stack([x_hw, torch.transpose(x_hw, dim0=2, dim1=3).contiguous()], dim=0).view(2, B, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[3])], dim=0) # (k, b, d, l)

        As = -torch.exp(self.A_logs.float())  # (4, inner, state)

        x_dbl = torch.einsum("k b d l, k c d -> k b c l", xs.view(4, B, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(4, 1, -1, 1) # (k, b, d, l)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("k b r l, k d r -> k b d l", dts.view(4, B, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(4, 1, -1, 1)

        out_y = batch_selective_scan_fn(
            xs.float(), 
            dts.float(), 
            As.float(), 
            Bs.float(), 
            Cs.float(), 
            self.Ds.float(), 
            None, 
            self.dt_projs_bias.float(),
            delta_softplus=True, 
            return_last_state=False, 
        )
        assert out_y.dtype == torch.float32

        inv_y = torch.flip(out_y[2:4], dims=[-1]).view(2, B, -1, L)
        wh_y = torch.transpose(out_y[1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[0] + inv_y[0] + wh_y + invwh_y
        y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        
        return out

    def forwardv2(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        x_hw = x
        x_hwwh = torch.stack([x_hw, torch.transpose(x_hw, dim0=2, dim1=3).contiguous()], dim=0).view(2, B, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[3])], dim=0) # (k, b, d, l)

        As = -torch.exp(self.A_logs.float())  # (4, inner, state)
        # Ds = self.Ds.float()
        # dt_projs_bias = self.dt_projs_bias.float()

        x_dbl = torch.einsum("k b d l, k c d -> k b c l", xs.view(4, B, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(4, 1, -1, 1) # (k, b, d, l)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("k b r l, k d r -> k b d l", dts.view(4, B, -1, L), self.dt_projs_weight)
        dts = dts + self.dt_projs_bias.float().view(4, 1, -1, 1)

        xs = xs.float()
        dts = dts.float()
        Bs = Bs.float()
        Cs = Cs.float()

        out_y = [None] * 4
        selective_scan_ = selective_scan_base
        # selective_scan_ = selective_scan_fn

        out_y[0] = selective_scan_(
                xs[0], dts[0], 
                As[0], Bs[0], Cs[0], None, z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
        out_y[1] = selective_scan_(
                xs[1], dts[1], 
                As[1], Bs[1], Cs[1], None, z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
        out_y[2] = selective_scan_(
                xs[2], dts[2], 
                As[2], Bs[2], Cs[2], None, z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
        out_y[3] = selective_scan_(
                xs[3], dts[3], 
                As[3], Bs[3], Cs[3], None, z=None,
                delta_bias=None,
                delta_softplus=True,
                return_last_state=False,
            )
        out_y = torch.stack(out_y) 

        # print(out_y.shape, self.Ds.shape, xs.shape)
        out_y = out_y + self.Ds.float().view(4, 1, -1, 1) * xs


        inv_y = torch.flip(out_y[2:4], dims=[-1]).view(2, B, -1, L)
        wh_y = torch.transpose(out_y[1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[0].float() + inv_y[0].float() + wh_y.float() + invwh_y.float()
        y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)

        return out


class BiMamba2DBlockSlow4(MambaBlock):
    ATTNBLOCK = BiMamba2DSlow4
    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += self.self_attention.flops(B=B, H=H, W=W)
        return flops
    

class BiMamba2DSlow1(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) 

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs)
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1) # (K=4, D, N)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs):
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
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        x_hw = x
        x_hwwh = torch.stack([x_hw, torch.transpose(x_hw, dim0=2, dim1=3).contiguous()], dim=0).view(2, B, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[3])], dim=0).view(4 * B, -1, L) # (kb, d, l)

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        dt_projs_bias = self.dt_projs.bias.float()

        x_dbl = torch.einsum("b d l, c d -> b c l", xs, self.x_proj.weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.einsum("b r l, d r -> b d l", dts, self.dt_projs.weight)

        y = selective_scan_fn(
                xs, dts, 
                As, Bs, Cs, Ds, z=None,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
                return_last_state=False,
            )
        y = y.view(4, B, -1, L)
        invhwwh_y = torch.flip(y[2:4], dims=[3])
        wh_y = torch.transpose(y[1].view(B, -1, W, H), dim0=2, dim1=3).contiguous()
        invwh_y = torch.transpose(invhwwh_y[1].view(B, -1, W, H), dim0=2, dim1=3).contiguous()

        y = y[0] + wh_y.view(B, -1, L) + invhwwh_y[0] + invwh_y.view(B, -1, L)
        y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class BiMamba2DBlockFast(MambaBlock):
    # ATTNBLOCK = BiMamba2DFast1 # params: 9315592
    ATTNBLOCK = BiMamba2DSlow4
    # ATTNBLOCK = BiMamba2DSlow1 # params: 93902344

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += self.self_attention.flops(B=B, H=H, W=W)
        return flops


# print(flops_selective_scan_ref(), flops_mamba())

# ================================
if __name__ == "__main__":
    import copy
    mod1 = BiMamba2DSlow4(d_model=92)
    mod2 = copy.deepcopy(mod1)
    mod1 = mod1.cuda()
    mod2 = mod2.cuda()

    x1 = torch.rand((128, 14, 14, 92)).cuda()
    x2 = x1.detach().clone()
    
    y1 = mod1.forward(x1)
    y2 = mod2.forwardv2(x2)
    print((y1 - y2).abs().sum())

    g1 = torch.autograd.grad(y1.sum(), mod1.parameters(), retain_graph=True)
    g2 = torch.autograd.grad(y2.sum(), mod2.parameters(), retain_graph=True)
    print(sum([(_g1 - _g2).abs().sum() for _g1, _g2 in zip(g1, g2)]))

    import time

    tim0 = time.time()
    time.sleep(2)
    
    for _ in range(100):
        _ = mod1.forward(x1)

    tim1 = time.time()
    time.sleep(2)

    for _ in range(100):
        _ = mod1.forwardv2(x1)

    tim2 = time.time()
    time.sleep(2)

    for _ in range(100):
        _ = mod2.forward(x1)

    tim3 = time.time()

    print(tim1-tim0, tim2-tim1, tim3-tim2)




