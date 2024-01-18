import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import LayerNorm, Module
import torch.utils.checkpoint as checkpoint
from torchvision.models.vision_transformer import VisionTransformer, MLPBlock, EncoderBlock
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref, mamba_inner_fn, mamba_inner_ref

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights, Mamba, create_block
from mamba_ssm.models.mixer_seq_simple import create_block as mamba_create_block, _init_weights
from mamba_ssm.models.mixer_seq_simple import Block as MBlock


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


# PatchEmbed with 2D output
class PatchEmbed2D(PatchEmbed):
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
            # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0, 2, 3, 1)  # B Ph, Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


# PatchEmbed with 2D output
class PatchMerging2D(PatchMerging):
    def forward(self, x):
        B, H, W, C = x.shape

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


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
    if False:
        ...
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
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
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
        last_state = None
        """
    
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if False:
        ...
        """
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
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """
    
    return flops

    flops = 0
    flops += 0
    flops += B * D * L * N
    flops += 2 * B * D * L * N
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


class ori_mambablock(MBlock):
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
        d_model=hidden_dim
        ssm_cfg=None
        norm_epsilon=0.00001
        rms_norm=False
        residual_in_fp32=True
        fused_add_norm=True
        layer_idx=None
        device=None # used in RMS
        dtype=None # used in RMS

        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        super().__init__(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.layer_idx = layer_idx

        # =============
        self.dim = hidden_dim
        self.input_resolution = input_resolution

    def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        return flops


class ori_mambablock_nofast(MBlock):
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
        d_model=hidden_dim
        ssm_cfg=dict(use_fast_path=False)
        norm_epsilon=0.00001
        rms_norm=False
        residual_in_fp32=True
        fused_add_norm=True
        layer_idx=None
        device=None # used in RMS
        dtype=None # used in RMS

        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        super().__init__(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.layer_idx = layer_idx

        # =============
        self.dim = hidden_dim
        self.input_resolution = input_resolution

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        return flops


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
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        input_resolution = (0, 0),
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim, use_fast_path=False)
        # self.dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(drop_path)

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
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn

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
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forwardv0(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        L = H * W

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        x_hw = x
        x_hwwh = torch.stack([x_hw, torch.transpose(x_hw, dim0=2, dim1=3).contiguous()], dim=0).view(2, B, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[3])], dim=0) # (k, b, d, l)

        As = -torch.exp(self.A_logs.float()).view(4, -1, self.d_state)  # (4, inner, state)
        Ds = self.Ds.float().view(4, -1)
        dt_projs_bias = self.dt_projs_bias.float().view(4, -1)

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

    def flopsv0(self, B=1, H=16, W=16):
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

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        L = H * W
        K = 4

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)

        x_hwwh = torch.stack([x, torch.transpose(x, dim0=2, dim1=3).contiguous()], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
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
        flops += flops_selective_scan_ref(B=B, L=L, D=d_inner * 4, N=d_state, with_D=True, with_Z=False) # y = selective_scan_fn( 
        flops += B * L * d_inner # y = self.out_norm(y)
        flops += B * L * d_inner * d_model    # out = self.out_proj(y)
        # print(flops, flops - 4 * flops_selective_scan_ref(B=B, L=L, D=d_inner, N=d_state, with_D=True, with_Z=False))
        return flops



# =======================

class BiMamba2DBlockSlow4(MambaBlock):
    ATTNBLOCK = BiMamba2DSlow4
    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += self.self_attention.flops(B=B, H=H, W=W)
        return flops
    

class BiMamba2DBlockSlow4Dp(nn.Module):
    ATTNBLOCK = BiMamba2DSlow4
    def __init__(
        self,
        num_heads: int = 0,
        hidden_dim: int = 0,
        mlp_dim: int = 0,
        dropout: float = 0,
        attention_dropout: float = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        input_resolution = (0, 0),
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = self.ATTNBLOCK(d_model=hidden_dim, use_fast_path=False)
        self.dropout = DropPath(drop_path)
        DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

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

        flops = 0
        flops += B * L * D  # norm
        flops += self.self_attention.flops(B=B, H=H, W=W)
        return flops


class ori_mambablock(MBlock):
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
        d_model=hidden_dim
        ssm_cfg=None
        norm_epsilon=0.00001
        rms_norm=False
        residual_in_fp32=True
        fused_add_norm=True
        layer_idx=None
        device=None # used in RMS
        dtype=None # used in RMS

        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        super().__init__(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.layer_idx = layer_idx

        # =============
        self.dim = hidden_dim
        self.input_resolution = input_resolution

    def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        block.layer_idx = layer_idx
        return block

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        return flops


class ori_mambablock_nofast(MBlock):
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
        d_model=hidden_dim
        ssm_cfg=dict(use_fast_path=False)
        norm_epsilon=0.00001
        rms_norm=False
        residual_in_fp32=True
        fused_add_norm=True
        layer_idx=None
        device=None # used in RMS
        dtype=None # used in RMS

        if ssm_cfg is None:
            ssm_cfg = {}
        factory_kwargs = {"device": device, "dtype": dtype}
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
        )
        super().__init__(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
        self.layer_idx = layer_idx

        # =============
        self.dim = hidden_dim
        self.input_resolution = input_resolution

    def flops(self):
        H, W = self.input_resolution
        B, L, D = 1, H * W, self.dim

        flops = 0
        flops += B * L * D  # norm
        flops += flops_mamba(B=B, L=L, d_model=D, 
            d_state=self.mixer.d_state, d_conv=self.mixer.d_conv, 
            d_inner=self.mixer.d_inner, dt_rank=self.mixer.dt_rank) # mixer == mamba
        return flops


# base of x, res = block(x, res)
class BasicLayerv0(nn.Module):
    MAMBABLOCK = ori_mambablock
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, **kwargs):
        # downsample = None
        initializer_cfg = None

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            self.MAMBABLOCK(
                hidden_dim=dim,
                mlp_dim = int(mlp_ratio * dim),
                dropout=drop,
                attention_dropout=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                input_resolution=self.input_resolution,
            )
            for i in range(depth)])
        
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        hidden_states = x # (128, 3136, 96)
        residual = None
        for blk in self.blocks:
            if self.use_checkpoint:
                hidden_states, residual = checkpoint.checkpoint(blk, hidden_states, residual)
            else:
                hidden_states, residual = blk(hidden_states, residual)
        x = hidden_states

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayerv1(BasicLayerv0):
    MAMBABLOCK = ori_mambablock_nofast


# base of x = block(x)
class BasicLayerv2(BasicLayerv0):
    MAMBABLOCK = MambaBlock

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        if self.downsample is not None:
            x = self.downsample(x)

        return x


# 1d base module
class VMamba(nn.Module):
    BASICLAYER = BasicLayerv0
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = self.BASICLAYER(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops

# 2d base module
class VMamba2D(nn.Module):
    class BASICLAYER(BasicLayerv2):
        MAMBABLOCK = BiMamba2DBlockSlow4

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed2D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = self.BASICLAYER(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               drop=drop_rate, 
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = torch.flatten(x, 1, 2) # B H W C -> B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class VMamba2Dp(VMamba2D):
    class BASICLAYER(BasicLayerv2):
        MAMBABLOCK = BiMamba2DBlockSlow4Dp


# =======================================
class cLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# the most seemingly-like nn.attention version 
class S62D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj_weight = nn.Parameter(torch.empty((self.d_inner * 2, self.d_model), **factory_kwargs))
        self.out_proj_weight = nn.Parameter(torch.empty((self.d_model, self.d_inner), **factory_kwargs))
        nn.init.xavier_normal_(self.in_proj_weight)
        nn.init.xavier_normal_(self.out_proj_weight)

        self.in_proj_bias = None
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty((self.d_inner * 2), **factory_kwargs))
            self.out_proj_bias = nn.Parameter(torch.empty((self.d_model), **factory_kwargs))
            nn.init.constant_(self.in_proj_bias)
            nn.init.constant_(self.out_proj_bias)

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.out_norm = cLayerNorm(self.d_inner, data_format="channels_first")

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
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor, **kwargs):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        io_bias = (self.in_proj_bias is not None)

        kv = torch.einsum("b c h w, d c -> b d h w", x, self.in_proj_weight)
        if io_bias:
            kv = kv + self.in_proj_bias.view(1, -1, 1, 1)
        key, value = kv.chunk(2, dim=1) # (b, c, h, w)

        x = key
        x_hwwh = torch.stack([x, torch.transpose(x, dim0=2, dim1=3).contiguous()], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = self.out_norm(y.view(B, -1, H, W)) * value

        out = torch.einsum("b d h w, c d -> b c h w", y, self.out_proj_weight)
        if io_bias:
            out = out + self.out_proj_bias.view(1, -1, 1, 1)

        return out

    def flops(self, B=1, H=16, W=16):
        L = H * W
        d_model = self.d_model
        d_inner = self.d_inner
        dt_rank = self.dt_rank
        d_state = self.d_state

        flops = 0
        flops += B * L * d_model * (d_inner * 2) # xz = ...
        flops += 4 * B * L * d_inner * (dt_rank + 2 * d_state) # x_dbl = ...
        flops += 4 * B * L * dt_rank * d_inner # dt =
        flops += flops_selective_scan_ref(B=B, L=L, D=d_inner * 4, N=d_state, with_D=True, with_Z=False) # y = selective_scan_fn( 
        flops += B * L * d_inner # y = self.out_norm(y)
        flops += B * L * d_inner * d_model    # out = self.out_proj(y)
        # print(flops, flops - 4 * flops_selective_scan_ref(B=B, L=L, D=d_inner, N=d_state, with_D=True, with_Z=False))
        return flops


def test_BiMamba2DSlow4():
    import copy, time
    mod1 = BiMamba2DSlow4(d_model=92)
    mod2 = copy.deepcopy(mod1)
    mod1 = mod1.cuda()
    mod2 = mod2.cuda()

    x1 = torch.rand((128, 14, 14, 92)).cuda()
    x2 = x1.detach().clone()
    
    y1 = mod1.forwardv0(x1)
    y2 = mod1.forward(x1)
    print((y1 - y2).abs().sum())

    g1 = torch.autograd.grad(y1.sum(), mod1.parameters(), retain_graph=True)
    g2 = torch.autograd.grad(y2.sum(), mod1.parameters(), retain_graph=True)
    print(sum([(_g1 - _g2).abs().sum() for _g1, _g2 in zip(g1, g2)]))

    tim0 = time.time()
    time.sleep(2)
    
    for _ in range(100):
        _ = mod1.forwardv0(x1)
        # _ = torch.transpose(x1, dim0=2, dim1=3)

    tim1 = time.time()
    time.sleep(2)

    for _ in range(100):
        _ = mod1.forward(x1)
        # _ = x1.permute(0, 1, 3, 2)

    tim2 = time.time()
    time.sleep(2)

    for _ in range(100):
        _ = mod2.forwardv0(x1)
        # _ = torch.transpose(x1, dim0=2, dim1=3)

    tim3 = time.time()

    print(tim1-tim0, tim2-tim1, tim3-tim2)


if __name__ == "__main__":
    test_BiMamba2DSlow4()


