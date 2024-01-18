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

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.models.mixer_seq_simple import _init_weights, Mamba, create_block
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref, mamba_inner_fn, mamba_inner_ref
from mamba_ssm.models.mixer_seq_simple import _init_weights, Mamba, create_block

# ===================
def flops_selective_scan_ref(B=1, L=256, D=768, N=16,  with_D=True, with_Z=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    """
    # delta = delta + delta_bias[..., None].float()
    # x = A.new_zeros((batch, dim, dstate))
    # deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    # deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
    # for i in range(u.shape[2]):
        # x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        # y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
    # out = y + u * rearrange(D, "d -> d 1")
    # out = out * F.silu(z)
    flops = 0
    flops += 0
    flops += B * D * L * N
    for _ in range(L):
        flops += B * D * N
        flops += B * D * N
    
    if with_D:
        flops += B * D * L

    if with_Z:
        flops += B * D * L
    
    return flops


def flops_mamba(B=1, L=256, d_model=92, d_state=16, d_conv=4, d_inner=184, dt_rank=6):
    """
    hidden_states: (B, L, D)
    """
    if True:
        ...
        # ==================================================================
        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # self.conv1d = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )

        # self.activation = "silu"
        # self.act = nn.SiLU()

        # self.x_proj = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        # self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # A = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_log = torch.log(A)  # Keep A_log in fp32
        # self.A_log = nn.Parameter(A_log)
        # self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        # ==================================================================

        # xz = rearrange(
        #     self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        #     "d (b l) -> b d l",
        #     l=seqlen,
        # )
        # A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
        #     ...
        # else:
        #     x, z = xz.chunk(2, dim=1)
        #     if causal_conv1d_fn is None:
        #         x = self.act(self.conv1d(x)[..., :seqlen])
        #     else:
        #         ...
        #     x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        #     dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        #     dt = self.dt_proj.weight @ dt.t()
        #     dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        #     B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        #     C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        #     assert self.activation in ["silu", "swish"]
        #     y = selective_scan_fn(
        #         x,
        #         dt,
        #         A,
        #         B,
        #         C,
        #         self.D.float(),
        #         z=z,
        #         delta_bias=self.dt_proj.bias.float(),
        #         delta_softplus=True,
        #         return_last_state=ssm_state is not None,
        #     )
        #     if ssm_state is not None:
        #         y, last_state = y
        #         ssm_state.copy_(last_state)
        #     y = rearrange(y, "b d l -> b l d")
        #     out = self.out_proj(y)
        # return out
    flops = 0
    flops += B * L * d_model * (d_inner * 2) # xz = ...
    cg = d_inner # conv group
    cgci, cgco = d_inner // cg, d_inner // cg # conv_group_channel_in, conv_group_channel_out
    flops += B * L * (cg * (cgci * cgco * d_conv)) # x = self.act(self.conv1d(x)[..., :seqlen])
    flops += B * L * d_inner * (dt_rank + 2 * d_state) # x_dbl = ...
    flops += B * L * dt_rank * d_inner # dt =
    flops += flops_selective_scan_ref(B=B, L=L, D=d_inner, N=d_state, with_D=True, with_Z=True) # y = selective_scan_fn( 
    flops += B * L * d_inner * d_model    # out = self.out_proj(y)
    return flops


# ===================


# same flops as original mixer_seq_simple.Block
class MambaBlock(nn.Module):
    ATTNBLOCK = Mamba
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int = 0,
        hidden_dim: int = 0,
        mlp_dim: int = 0,
        dropout: float = 0,
        attention_dropout: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        input_resolution = (0, 0),
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim, use_fast_path=False)
        self.dropout = nn.Dropout(dropout)

        # =============
        self.dim = hidden_dim
        self.input_resolution = input_resolution

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x = self.self_attention(x)
        x = self.dropout(x)
        x = x + input
        return x

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim
        self.mixer = self.ATTNBLOCK

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        return flops


# same flops as MambaBlock
class MambaBlockv0(nn.Module):
    ATTNBLOCK = Mamba

    def __init__(
        self,
        num_heads: int = 0,
        hidden_dim: int = 0,
        mlp_dim: int = 0,
        dropout: float = 0,
        attention_dropout: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        input_resolution = (0, 0),
        **kwargs,
    ):
        super().__init__()
        self.norm: nn.LayerNorm = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim, use_fast_path=False)
        self.dropout = nn.Dropout(dropout)
        self.fused_add_norm = False
        self.residual_in_fp32 = True

        # =============
        self.dim = hidden_dim
        self.input_resolution = input_resolution

    def forward(self, input: torch.Tensor, residual: torch.Tensor = None):
        hidden_states = input
        
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )

        x = self.self_attention(hidden_states)
        x = self.dropout(x)
        return x, residual

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim
        self.mixer = self.ATTNBLOCK

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        return flops


# same flops as MambaBlock + B * L * D
class MambaBlockv1(nn.Module):
    ATTNBLOCK = Mamba

    def __init__(
        self,
        num_heads: int = 0,
        hidden_dim: int = 0,
        mlp_dim: int = 0,
        dropout: float = 0,
        attention_dropout: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim,use_fast_path=False)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x = self.self_attention(x)
        if self.gamma:
            x = self.gamma * x
        x = self.dropout(x)
        x = x + input
        return x

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim
        self.mixer = self.ATTNBLOCK

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        
        if self.gamma:
            flops += B * L * D

        return flops


# ===========================================
class BiMamba(nn.Module):
    def __init__(self, *args,**kwargs):
        super().__init__()
        self.mamba = Mamba(*args, **kwargs)
        self.invmamba = Mamba(*args, **kwargs)

    def forward(self, x):
        y = self.mamba(x)
        iy = self.invmamba(torch.flip(x, dims=[1]))
        return y + iy


class BiMambaBlock(MambaBlock):
    ATTNBLOCK = BiMamba


# ===========================================
def simp_test():
    import torch
    from einops import rearrange, einsum
    B, D, L, R = 3, 4, 5, 6
    dt = torch.randn(((B*L), R))
    dtp = torch.randn((D, R))
    xxx = rearrange(dtp @ dt.t(), "d (b l) -> b d l", l=L)
    dr = rearrange(dt, "(b l) r -> b r l", l=L)
    yyy = torch.einsum("b r l, d r -> b d l", dr, dtp)
    print((xxx - yyy).abs().sum())


class BiMambav3(nn.Module):
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

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Conv1d(
            self.d_inner, 2 * (self.dt_rank + self.d_state * 2), kernel_size=1, stride=1, padding=0, bias=False, **factory_kwargs
        )
        self.dt_projs = nn.ModuleList((
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
        ))
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2)
        self.Ds = self.D_init(self.d_inner, copies=2)

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

    def forward(self, x, **kwargs):
        batch, seqlen, dim = x.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x))
        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(x)  # (b, d, l)
        dts, BCs = torch.split(x_dbl, [self.dt_rank * 2, self.d_state * 4], dim=1)
        BCs = torch.split(BCs, self.d_state, dim=1) # (B1, B2, ..., C1, C2,...)
        dts = torch.split(dts, self.dt_rank, dim=1) # (dt1, dt2,...)
        dts = [torch.einsum("b r l, d r -> b d l", t, self.dt_projs[i].weight) for i, t in enumerate(dts)]
        As = -torch.exp(self.A_logs.float())  # (2, d_inner, d_state)
        Bs = BCs[0:2] # (2, b, d_state, l)
        Cs = BCs[2:4] # (2, b, d_state, l)
        Ds = self.Ds.float()
        xs = [x, torch.flip(x, dims=[1])]
        
        ys = []
        for i in range(2):
            ys.append(selective_scan_fn(
                xs[i], dts[i], 
                As[i], Bs[i], Cs[i], Ds[i], z=None,
                delta_bias=self.dt_projs[i].bias.float(),
                delta_softplus=True,
                return_last_state=False,
            ))
        y = sum(ys)
        y = y * F.silu(z)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


class BiMambaBlockv3(MambaBlock):
    ATTNBLOCK = BiMambav3


# ===========================================

class BiMamba2D(nn.Module):
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
        self.h_bimamba = BiMamba(**kwargs)
        self.w_bimamba = BiMamba(**kwargs)

    def forward(self, x):
        B = x.shape[0]
        xw = rearrange(x, "b h w c -> (b w) h c")
        xw = self.w_bimamba(xw)
        xh = rearrange(xw, "(b w) h c -> (b h) w c", b=B)
        xh = self.h_bimamba(xh)
        y = rearrange(xh, "(b h) w c -> b h w c", b=B)
        return y


class BiMamba2DBlock(MambaBlock):
    ATTNBLOCK = BiMamba2D


# ===========================================

# failed
class BiMamba2Dv2(nn.Module):
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
        self.h_bimamba = BiMamba(**kwargs)
        self.norm = nn.LayerNorm(d_model)
        self.w_bimamba = BiMamba(**kwargs)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xw = rearrange(x, "b h w c -> b (w h) c")
        xw = self.w_bimamba(xw)
        xh = rearrange(xw, "b (w h) c -> b (h w) c", h=H)
        xh = self.norm(xh) + x.view(B, -1, C)
        xh = self.h_bimamba(xh)
        y = rearrange(xh, "b (h w) c -> b h w c", h=H)
        return y


class BiMamba2DBlockv2(MambaBlock):
    ATTNBLOCK = BiMamba2Dv2



# ===========================================
class BiMamba2Dv3(nn.Module):
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

        self.x_proj = nn.Conv2d(
            self.d_inner, 4 * (self.dt_rank + self.d_state * 2), kernel_size=1, stride=1, padding=0, bias=False, **factory_kwargs
        )
        self.dt_projs = nn.ModuleList((
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
        ))
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4)
        self.Ds = self.D_init(self.d_inner, copies=4)

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

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(x)  # (b, d, h, w)
        hw_param, wh_param = torch.split(x_dbl, x_dbl.shape[1] // 2, dim=1)
        wh_param = wh_param.permute(0, 1, 3, 2).contiguous()
        hw_dts, hw_BCs = torch.split(hw_param.flatten(2, -1), [self.dt_rank * 2, self.d_state * 4], dim=1)
        wh_dts, wh_BCs = torch.split(wh_param.flatten(2, -1), [self.dt_rank * 2, self.d_state * 4], dim=1)
        hw_BCs = torch.split(hw_BCs, self.d_state, dim=1) # (B1, B2,..., C1, C2,...)
        wh_BCs = torch.split(wh_BCs, self.d_state, dim=1) # (B1, B2,..., C1, C2,...)
        hw_dts = torch.split(hw_dts, self.dt_rank, dim=1) # (dt1, dt2,...)
        wh_dts = torch.split(wh_dts, self.dt_rank, dim=1) # (dt1, dt2,...)
        dts = [torch.einsum("b r l, d r -> b d l", t, self.dt_projs[i].weight) for i, t in enumerate([*hw_dts, *wh_dts])]
        As = -torch.exp(self.A_logs.float())  # (4, d_inner, d_state)
        Bs = [*hw_BCs[0:2], *wh_BCs[0:2]] # (4, b, d_state, l)
        Cs = [*hw_BCs[2:4], *wh_BCs[2:4]] # (4, b, d_state, l)
        Ds = self.Ds.float()
        
        hw_x = x.flatten(2, -1) # (b, d, l)
        wh_x = x.permute(0, 1, 3, 2).contiguous().flatten(2, -1)
        xs = [hw_x, torch.flip(hw_x, dims=[1]), wh_x, torch.flip(wh_x, dims=[1])]
        
        ys = []
        for i in range(4):
            ys.append(selective_scan_fn(
                xs[i], dts[i], 
                As[i], Bs[i], Cs[i], Ds[i], z=None,
                delta_bias=self.dt_projs[i].bias.float(),
                delta_softplus=True,
                return_last_state=False,
            ))
        y = sum(ys)
        y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class BiMamba2DBlockv3(MambaBlock):
    ATTNBLOCK = BiMamba2Dv3


class BiMamba2DBlockv3a(MambaBlockv0):
    ATTNBLOCK = BiMamba2Dv3


class BiMamba2DBlockv3b(MambaBlockv1):
    ATTNBLOCK = BiMamba2Dv3


# ===========================================
    
class BiMamba2Dv4(nn.Module):
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
        self.wmamba = BiMambav3(**kwargs)
        self.hmamba = BiMambav3(**kwargs)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        x = self.wmamba(x.view(B * H, W, C))
        x = x.view(B, H, W, C).permute(0, 2, 1, 3).contiguous()
        x = self.hmamba(x.view(B * W, H, C))
        x = x.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        return x


class BiMamba2DBlockv4(MambaBlock):
    ATTNBLOCK = BiMamba2Dv4


# ============================================
class BiMambav4(nn.Module):
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

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Conv1d(
            self.d_inner, 2 * (self.dt_rank + self.d_state * 2), kernel_size=1, stride=1, padding=0, bias=False, **factory_kwargs
        )
        self.dt_projs = nn.ModuleList((
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, factory_kwargs),
        ))
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2)
        self.Ds = self.D_init(self.d_inner, copies=2)

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

    def forward(self, x, **kwargs):
        batch, seqlen, dim = x.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        x = self.act(self.conv1d(x))
        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(x)  # (b, d, l)
        dts, BCs = torch.split(x_dbl, [self.dt_rank * 2, self.d_state * 4], dim=1)
        BCs = torch.split(BCs, self.d_state, dim=1) # (B1, B2, ..., C1, C2,...)
        dts = torch.split(dts, self.dt_rank, dim=1) # (dt1, dt2,...)
        dts = [torch.einsum("b r l, d r -> b d l", t, self.dt_projs[i].weight) for i, t in enumerate(dts)]
        As = -torch.exp(self.A_logs.float())  # (2, d_inner, d_state)
        Bs = BCs[0:2] # (2, b, d_state, l)
        Cs = BCs[2:4] # (2, b, d_state, l)
        Ds = self.Ds.float()
        xs = [x, torch.flip(x, dims=[1])]
        
        ys = []
        for i in range(2):
            ys.append(selective_scan_fn(
                xs[i], dts[i], 
                As[i], Bs[i], Cs[i], Ds[i], z=None,
                delta_bias=self.dt_projs[i].bias.float(),
                delta_softplus=True,
                return_last_state=False,
            ))
        y = sum(ys)

        y = rearrange(y, "b d l -> b l d")
        z = rearrange(z, "b d l -> b l d")
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class BiMamba2Dv5(nn.Module):
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
        
        super().__init__()
        self.wmamba = BiMambav4(**kwargs)
        self.hmamba = BiMambav4(**kwargs)

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        x = self.wmamba(x.view(B * H, W, C))
        x = x.view(B, H, W, C).permute(0, 2, 1, 3).contiguous()
        x = self.hmamba(x.view(B * W, H, C))
        x = x.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        return x


class BiMamba2DBlockv5(MambaBlock):
    ATTNBLOCK = BiMamba2Dv5


def test_fused_add_norm_fn():
    hidden_states = torch.randn((10, 11, 12, 13)).cuda()
    ori_hidden_states = hidden_states.clone()
    residual = None
    norm = nn.LayerNorm(13).cuda()
    # ==============================================
    residual = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states = norm(residual.to(dtype=norm.weight.dtype))
    residual = residual.to(torch.float32)
    x, res = hidden_states.clone(), residual.clone()
    # ==============================================
    hidden_states = ori_hidden_states
    residual = None
    residual_in_fp32 = True
    fused_add_norm_fn = rms_norm_fn if isinstance(norm, RMSNorm) else layer_norm_fn
    hidden_states, residual = fused_add_norm_fn(
        hidden_states,
        norm.weight,
        norm.bias,
        residual=residual,
        prenorm=True,
        residual_in_fp32=residual_in_fp32,
        eps=norm.eps,
    )

    print((x -  hidden_states).abs().sum(), (res - residual).abs().sum())


if __name__ == "__main__":
    test_fused_add_norm_fn()