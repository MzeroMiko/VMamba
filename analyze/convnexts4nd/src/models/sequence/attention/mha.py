"""Wrapper around nn.MultiheadAttention to adhere to SequenceModule interface."""

import torch
import torch.nn.functional as F
from torch import nn
import hydra
from models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U
from einops import rearrange

@TransposedModule
class MultiheadAttention(SequenceModule):
    """Simple wrapper for MultiheadAttention."""
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.mha = nn.MultiheadAttention(d_model, n_heads, *args, batch_first=True, **kwargs)
        self.causal = causal

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        if self.causal and attn_mask is None:
            attn_mask = torch.triu(torch.ones(src.size(-2), src.size(-2),
                                              dtype=torch.bool, device=src.device),
                                       diagonal=1)
        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        y, _ = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return y, None

    def step(self, x, state):
        # TODO proper cached inference
        # x: (B, D)
        y, z = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, **kwargs)


class VitAttention(SequenceModule):
    """Copied from implementation for ViT: only used for ViT model.

    This attention class makes several simplifying assumptions (commonly satisfied in vision
       applications):
    1. q = k = v
    2. No masks: no attention mask, no key padding mask
    3. Embed dimension = Input dimension, i.e. projection matrices are square.

    Arguments:
    - packed_linear: whether to pack all 3 q_proj, k_proj, v_proj into 2 matrix.
        This option is to be compatible with T2T-ViT pretrained weights,
        where there's only one projection weight matrix.
    """

    @property
    def d_output(self):
        return self.dim

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        # proj_drop=0.,
        packed_linear=True,
        linear_cfg=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if linear_cfg is not None:
            packed_linear = False
        self.packed_linear = packed_linear
        if packed_linear:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            if linear_cfg is None:
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.q_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.k_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.v_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        # Removing this dropout because we do this in SequenceResidualBlock
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, state=None):
        B, N, C = x.shape
        if self.packed_linear:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]

        # attn = (q @ k.transpose(-2, -1) * self.scale)
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = q.size()
        _, _, k_seq_len, _ = k.size()
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h s d -> (b h) d s')
        # Preallocate attn_weights for `baddbmm`
        attn = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=q.dtype, device=q.device)
        attn = rearrange(torch.baddbmm(attn, q, k, beta=0, alpha=self.scale),
                         '(b h) t s -> b h t s', h = self.num_heads)

        attn = F.softmax(attn, dim=-1, dtype=v.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x, None
