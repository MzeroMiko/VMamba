import math
from functools import partial

from collections import namedtuple

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time


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

# =================================================================
from typing import Callable
from mamba_ssm.models.mixer_seq_simple import create_block as mamba_create_block, _init_weights
# https://huggingface.co/state-spaces/mamba-2.8b/blob/main/config.json
from .vs6 import MambaBlock, BiMambaBlock, BiMamba2DBlock, BiMamba2DBlockv2
from .vs6 import BiMambaBlockv3, BiMamba2DBlockv3, BiMamba2DBlockv3a, BiMamba2DBlockv4, BiMamba2DBlockv5

from mamba_ssm.models.mixer_seq_simple import Block as MBlock
from .vs6v2 import flops_selective_scan_ref, flops_mamba

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


# =========================================
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


class BasicLayerv3(BasicLayerv2):
    MAMBABLOCK = BiMambaBlock


class BasicLayerv4(BasicLayerv2):
    MAMBABLOCK = BiMambaBlockv3


# =========================================

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


# base + no fast path
class VMambav1(VMamba):
    BASICLAYER = BasicLayerv1


# MambaBlock # vit block version of mamba
class VMambav2(VMamba):
    BASICLAYER = BasicLayerv2
   

# BliMambaBlock # mamba + (flip + mamba)
class VMambav3(VMamba):
    BASICLAYER = BasicLayerv3


# BiMambaBlockv3 # mamba(s6 + (flip + s6))
class VMambav4(VMamba):
    BASICLAYER = BasicLayerv4


# =========================================

# =========================================
# PatchEmbed with 2D output
class PatchEmbed2D(PatchEmbed):
    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).permute(0, 2, 3, 1)  # B Ph, Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


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


# =========================================
# 2d base module
class VMamba2D(nn.Module):
    class BASICLAYER(BasicLayerv2):
        MAMBABLOCK = BiMamba2DBlockv3

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
                               drop=drop_rate, attn_drop=attn_drop_rate,
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


class VMamba2Da(VMamba2D):
    class BASICLAYER(BasicLayerv0):
        MAMBABLOCK = BiMamba2DBlockv3a


class VMamba2Db(VMamba2D):
    class BASICLAYER(BasicLayerv0):
        MAMBABLOCK = BiMamba2DBlockv4


class VMamba2Dc(VMamba2D):
    class BASICLAYER(BasicLayerv2):
        MAMBABLOCK = BiMamba2DBlockv5

# ===========================================================
# all based version2
from .vs6v2 import MambaBlockRef0, MambaBlockRef1, MambaBlockRef2, BiMambaBlockFast, BiMamba2DBlockFast
from .vs6v2 import BiMamba2DBlockSlow4

# class VMambaRef(VMamba):
class VMambaRef(VMamba2D):
    class BASICLAYER(BasicLayerv2):
        # MAMBABLOCK = MambaBlockRef0
        # MAMBABLOCK = MambaBlockRef1
        # MAMBABLOCK = MambaBlockRef2
        # MAMBABLOCK = BiMambaBlockFast
        MAMBABLOCK = BiMamba2DBlockFast

class VMamba2Dn1Slow4(VMamba2D):
    class BASICLAYER(BasicLayerv2):
        MAMBABLOCK = BiMamba2DBlockSlow4

VMambaRef = VMamba2Dn1Slow4

# ===========================================================
from .vs6 import BiMamba2Dv3, selective_scan_fn, F

class BiMamba2Dv3_for_log(BiMamba2Dv3):
    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        # print("????????????????")

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

        if True:
            is_nan = torch.isnan(As).any()
            is_inf = torch.isinf(As).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}mamba As, is nan {is_nan} inf {is_inf} after ssm", flush=True)
            is_nan = torch.isnan(Ds).any()
            is_inf = torch.isinf(Ds).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}mamba Ds, is nan {is_nan} inf {is_inf} after ssm", flush=True)

        
        hw_x = x.flatten(2, -1) # (b, d, l)
        wh_x = x.permute(0, 1, 3, 2).contiguous().flatten(2, -1)
        xs = [hw_x, torch.flip(hw_x, dims=[1]), wh_x, torch.flip(wh_x, dims=[1])]
        
        if True:
            is_nan = torch.isnan(hw_x).any()
            is_inf = torch.isinf(hw_x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}mamba hw_x, is nan {is_nan} inf {is_inf} after ssm", flush=True)

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

        if True:
            is_nan = torch.isnan(y).any()
            is_inf = torch.isinf(y).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}mamba y, is nan {is_nan} inf {is_inf} after ssm", flush=True)


        y = y.permute(0, 2, 1).contiguous().view(B, H, W, -1)
        yz = y * F.silu(z)
        out = self.out_proj(yz)

        if True:
            is_nan = torch.isnan(yz).any()
            is_inf = torch.isinf(yz).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}mamba yz, is nan {is_nan} inf {is_inf} after ssm", flush=True)
                print(f"cuda {torch.distributed.get_rank()}mamba yz, y {y.abs().max()} z {z.abs().max()} x {hw_x.abs().max()} A {As.abs().max()} D {Ds.abs().max()}", flush=True)

        if True:
            is_nan = torch.isnan(out).any()
            is_inf = torch.isinf(out).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}mamba out, is nan {is_nan} inf {is_inf} after ssm", flush=True)

        return out


class BiMamba2DBlockv3_for_log(BiMamba2DBlockv3):
    ATTNBLOCK = BiMamba2Dv3_for_log

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 4, f"Expected (batch_size, height, width, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        
        if True:
            is_nan = torch.isnan(x).any()
            is_inf = torch.isinf(x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}ln_1, is nan {is_nan} inf {is_inf} between layer", flush=True)

        x = self.self_attention(x)

        if True:
            is_nan = torch.isnan(x).any()
            is_inf = torch.isinf(x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}attn, is nan {is_nan} inf {is_inf} between layer", flush=True)

        x = self.dropout(x)

        if True:
            is_nan = torch.isnan(x).any()
            is_inf = torch.isinf(x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}drop, is nan {is_nan} inf {is_inf} between layer", flush=True)

        x = x + input

        if True:
            is_nan = torch.isnan(x).any()
            is_inf = torch.isinf(x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}add, is nan {is_nan} inf {is_inf} between layer", flush=True)

        return x


class BasicLayerv5_for_log(BasicLayerv2):
    MAMBABLOCK = BiMamba2DBlockv3_for_log

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            
            is_nan = torch.isnan(x).any()
            is_inf = torch.isinf(x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}block {i}, is nan {is_nan} inf {is_inf} between layer", flush=True)

        
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class vmamba2d_for_log(VMamba2D):
    BASICLAYER = BasicLayerv5_for_log
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        is_nan = torch.isnan(x).any()
        is_inf = torch.isinf(x).any()
        if is_nan or is_inf:
            print(f"cuda {torch.distributed.get_rank()}-1, is nan {is_nan} inf {is_inf} before layer", flush=True)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            is_nan = torch.isnan(x).any()
            is_inf = torch.isinf(x).any()
            if is_nan or is_inf:
                print(f"cuda {torch.distributed.get_rank()}layer {i}, is nan {is_nan} inf {is_inf} between layer", flush=True)

        x = torch.flatten(x, 1, 2) # B H W C -> B L C
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x


VMamba2Db = vmamba2d_for_log  

