# Adapted from https://github.com/facebookresearch/mega/blob/ea355255149d38ffe16bf2c176d47c3864e8b05a/fairseq/modules/moving_average_gated_attention.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Mega block design (Gating + Attention + EMA/SSM)."""

import math
from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.nn import Activation, DropoutNd, Normalization
from src.models.sequence.backbones.block import SequenceResidualBlock
from src.models.sequence.kernels.fftconv import FFTConv


class MegaBlock(nn.Module):
    """Block design from "Mega: Exponential Moving Average Gated Attention" paper.

    This class is a consolidated version of the MovingAveragedGatedAttention and MegaEncoderLayer classes
    from the official Mega code. They have been consolidated into one class, combining the EMA+Attention
    module together with the feed-forward network (FFN) module, by composing primitives from this codebase.
    This is meant to be a faithful adaptation of the original code, with the following changes:
    - Several variable names have been changed to be consistent with this codebase.
    - Some annotations have been changed and added, referencing the original paper and code where possible.
    - The recurrent state implementation has been removed, which had a design pattern that departs
      too much from this codebase. An adaptation of this functionality may be added in the future.

    An additional feature is supported where aside from the original MultiHeadEMA,
    other forms of global convolution from models.sequence.kernels can be used interchangeably.
    """

    def __init__(
        # Options are annotated with the original argument names
        # from MovingAverageGatedAttention and MegaEncoderLayer
        self,
        d_model,           # Mega: embed_dim
        d_attin,           # Mega: zdim
        d_attout,          # Mega: hdim
        d_state,           # Mega: ndim
        dropout=0.0,
        drop_attin=None,   # Mega: attention_dropout
        drop_attout=None,  # Mega: hidden_dropout
        activation='silu',
        attention_activation='softmax',
        bidirectional=False,
        chunk=-1,          # Mega: chunk_size
        l_max=None,        # Mega: truncation
        norm='layer',      # Mega: norm_type
        prenorm=True,
        tie_dropout=False, # Mega: feature_dropout
        rel_pos_bias='simple',
        max_positions=1024,
        ff_expand=2,       # Mega: encoder_ffn_embed_dim
        drop_ffn=None,     # Mega: activation_dropout
        transposed=False,  # Inputs shape (B L D)
        mode='mega',
        # If mode='mega', use the official Mega MultiHeadEMA class verbatim
        # Otherwise, construct a convolution kernel from kernel.py and use a general SSM wrapper
        # mode='ema' uses the same core kernel code from MultiHeadEMA, and should be exactly the same as mode='mega'
        # mode='nplr' uses the S4 kernel
        # mode='diag' uses the S4D kernel, etc.
        **ssm_args, # pass other keyword arguments to the SSM kernels
    ):
        super().__init__()
        self.transposed = transposed
        self.d_model = d_model
        self.d_output = d_model

        self.d_attout = d_attout
        self.d_attin = d_attin
        self.d_state = d_state
        self.activation = Activation(activation)
        self.attention_activation_fn = None if attention_activation == 'softmax' else Activation(attention_activation)
        self.scaling = self.d_attin ** -0.5 if attention_activation == 'softmax' else None


        # Configure dropout
        if drop_attin is None: drop_attin = dropout
        if drop_attout is None: drop_attout = dropout
        if drop_ffn is None: drop_ffn = dropout
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_attout = dropout_fn(drop_attout) if drop_attout > 0.0 else nn.Identity()
        self.drop_attin = nn.Dropout(drop_attin)

        self.l_chunk = chunk
        self.prenorm = prenorm
        self.norm = Normalization(d_model, _name_=norm, transposed=False)

        # Construct a *linear* SSM
        if mode == 'mega':
            self.ssm = MultiHeadEMA(
                d_model,
                d_state=d_state,
                bidirectional=bidirectional,
                l_max=l_max,
            )
        else:
            self.ssm = FFTConv(
                d_model,
                d_state=d_state,
                bidirectional=bidirectional,
                l_max=l_max,
                activation=None,
                mode=mode,
                transposed=False,
                **ssm_args,
            )

        self.v_proj = nn.Linear(d_model, d_attout)  # U_v (eq. 10)
        self.mx_proj = nn.Linear(d_model, d_attin + d_attout + 2 * d_model)
        self.h_proj = nn.Linear(d_attout, d_model)  # U_h (eq. 14)

        self.gamma = nn.Parameter(torch.Tensor(2, d_attin))
        self.beta = nn.Parameter(torch.Tensor(2, d_attin))

        self.max_positions = max_positions
        max_positions = max_positions if self.l_chunk < 0 else self.l_chunk
        if rel_pos_bias == 'simple':
            self.rel_pos_bias = SimpleRelativePositionalBias(max_positions)
        elif rel_pos_bias == 'rotary':
            self.rel_pos_bias = RotaryRelativePositionalBias(d_attin, max_positions)
        else:
            raise ValueError('unknown relative position bias: {}'.format(rel_pos_bias))

        # NFFN (normalized feed-forward network)
        if ff_expand is not None and ff_expand > 0:
            ffn_cfg = {
                '_name_': 'ffn',
                'expand': ff_expand,
                'activation': activation,
                'dropout': drop_ffn,
                'tie_dropout': tie_dropout,
                'transposed': transposed,
            }
            self.nffn = SequenceResidualBlock(
                d_model,
                prenorm=prenorm,
                dropout=dropout,
                tie_dropout=tie_dropout,
                residual='R',
                norm=norm,
                layer=ffn_cfg,
                transposed=transposed,
            )
        else:
            self.nffn = None

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        if padding_mask is not None:
            inverse_mask = 1.0 - padding_mask.type_as(q)      # (B K C)
            lengths = inverse_mask.sum(dim=-1, keepdim=True)  # (B K 1)
            lengths = lengths.clamp(min=1.0).unsqueeze(-1)    # (B K 1 1)  TODO finish transcribing
        else:
            lengths = slen
            inverse_mask = None

        if attn_mask is not None:
            # C x 1
            lengths = attn_mask.sum(dim=-1, keepdim=True)

        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) / lengths + bias

        if before_attn_fn:
            return qk

        attn_weights = self.attention_activation_fn(qk)

        if inverse_mask is not None:
            attn_weights = attn_weights * inverse_mask.unsqueeze(2)

        if attn_mask is not None:
            attn_weights = attn_weights * attn_mask

        return attn_weights

    def softmax_attention(self, q, k, padding_mask, attn_mask, before_attn_fn):
        slen = k.size(2)
        # C x C
        bias = self.rel_pos_bias(slen)
        if slen != q.size(2):
            assert q.size(2) == 1
            # 1 x C
            bias = bias[-1:]

        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) + bias

        if attn_mask is not None:
            qk = qk + attn_mask

        if padding_mask is not None:
            padding_mask_all = padding_mask.all(dim=-1, keepdim=True)
            padding_mask = torch.logical_and(padding_mask, ~padding_mask_all)
            qk = qk.masked_fill(padding_mask.unsqueeze(2).to(torch.bool), float('-inf'))

        if before_attn_fn:
            return qk

        attn_weights = F.softmax(qk, dim=-1)
        return attn_weights

    def forward(
        self,
        x,
        state=None, # TODO consolidate with incremental_state
        padding_mask: Optional[torch.Tensor] = None,  # Mega: encoder_padding_mask
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        before_attn_fn: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: (B L D) = batch, length, dimension

        Dimensions:
            B: Batch size
            L: Sequence length (l_seq)
            C: Chunk size
            K: Number of chunks (L / C)
            D: Model dimension (d_model)     ($d$ in paper)
            V: Dim. attention output (Mega: paper $v$, code `d_attout`, annotation E)
            Z: Dim. attention input  (Mega: paper $z$, code `d_attin`, annotation S)
        """
        if self.transposed: x = x.transpose(-1, -2)  # (B L D)
        B, L, D = x.size()
        assert D == self.d_model

        residual = x
        if self.prenorm:
            x = self.norm(x)

        v = self.activation(self.v_proj(x))  # (B L V)

        mx, _ = self.ssm(x, state=state, padding_mask=padding_mask)  # (B L D)
        # Original Mega code bakes a SiLU activation at the end of the MultiHeadEMA module
        # It has been moved here, which makes more sense to keep the SSM module linear
        # and so the activation is configurable and consistent with the rest of the block
        mx = self.activation(mx)
        mx = self.dropout(mx)

        base = self.mx_proj(mx)  # (B L D) -> (B L 2*D+Z+V)
        u, zr, hx = torch.split(base, [D, self.d_attin + self.d_attout, D], dim=-1)
        u = torch.sigmoid(u)  # (B L D) \phi (eq. 13)
        # Mega specifies to hard-code SiLU here, but the self.activation is always silu
        # in their configs anyways so this seems more sensible in case it's changed
        z, r = torch.split(self.activation(zr), [
            self.d_attin,  # z = (B L Z) Z (eq. 7)
            self.d_attout, # r = (B L V) \gamma (eq. 12)
        ], dim=-1)
        z = z.unsqueeze(2) * self.gamma + self.beta
        q, k = torch.unbind(z, dim=2)  # (B L Z) Q and K (eq. 8 and 9)

        q = q.unsqueeze(1)  # (B 1 L Z)
        k = k.unsqueeze(1)  # (B 1 L Z)
        v = v.unsqueeze(1)  # (B 1 L Z)
        if self.l_chunk < 0:
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1)  # (B 1 L)
        else:
            if L < self.l_chunk:
                pass
            else:
                q = rearrange(q, 'b 1 (k c) z -> b k c z', c=self.l_chunk)

            l_ctx = k.size(2)  # Transcribed from orig, why is this not the same as L?
            if l_ctx < self.l_chunk:
                if padding_mask is not None:
                    padding_mask = padding_mask.unsqueeze(1)  # (B 1 C)?
            else:
                k = rearrange(k, 'b 1 (k c) z -> b k c z', c=self.l_chunk)
                v = rearrange(v, 'b 1 (k c) z -> b k c z', c=self.l_chunk)
                if padding_mask is not None:
                    padding_mask = rearrange(padding_mask, 'b (k c) -> b k c', c=self.l_chunk)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if padding_mask is not None and padding_mask.dim() == 0:
            padding_mask = None

        if self.attention_activation_fn is None:  # Softmax case
            attn_weights = self.softmax_attention(q, k, padding_mask, attn_mask, before_attn_fn)
        else:
            attn_weights = self.element_attention(q, k, padding_mask, attn_mask, before_attn_fn)

        if before_attn_fn:
            if self.transposed: v = v.transpose(-1, -2)
            # return attn_weights, v  # TODO looks like bug in orig code
            return v, attn_weights

        v = self.drop_attout(v)  # (B K C V)
        kernel = self.drop_attin(attn_weights)  # (B K C C)
        h = rearrange(torch.matmul(kernel, v), 'b k c v -> b (k c) v') # (B L V)
        h = self.activation(hx + self.h_proj(h * r))  # (B L D)
        h = self.dropout(h)
        # Output (y) from update gate u (\phi): u * h + (1-u) * x, eq. (15)
        out = torch.addcmul(residual, u, h - residual)  # (B L D)

        if not self.prenorm:
            out = self.norm(out)

        if self.transposed: out = out.transpose(-1, -2)

        # FFN
        out, _ = self.nffn(out, state=None)


        if not need_weights: attn_weights = None
        # Because this class expects to return a state, it's a little inconvenient to return attention weights.
        # The official Mega code doesn't return it either.
        return out, _ # , attn_weights

    def extra_repr(self) -> str:
        return 'd_model={}, d_attin={}, d_attout={}, d_state={}, chunk={}, attn_act={}, prenorm={}'.format(self.d_model, self.d_attin,
                                                                                  self.d_attout, self.d_state, self.l_chunk,
                                                                                  self.attention_activation, self.prenorm)

"""
EMA (Exponential Moving Average) module.

Adapted from https://github.com/facebookresearch/mega/blob/ea355255149d38ffe16bf2c176d47c3864e8b05a/fairseq/modules/exponential_moving_average.py
"""
class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    This class is a verbatim translation of the original code with minor differences that
    do not change the code execution path.

    - A few variable names have been changed to be more consistent with this codebase.

    - State passing is not supported ("incremental_state" in the Mega code),
      as the original module uses a different fairseq interface than this codebase.

    - The only semantic change is removing the final SiLU activation,
      which is handled by the caller module (e.g. src.models.sequence.mega.MegaBlock).

    It is possible to recreate the functionality of MultiHeadEMA by using other modular blocks,
    in particular the src.models.sequence.kernels.fftconv.FFTConv block
    together with src.models.sequence.kernels.kernel.EMAKernel.
    """

    def __init__(
        self,
        d_model,
        d_state=2,
        bidirectional=False,
        l_max=None,
    ):
        super().__init__()

        self.H = d_model
        self.N = d_state
        self.bidirectional = bidirectional
        self.l_max = l_max
        self.scale = math.sqrt(1.0 / self.N)

        H = 2 * self.H if self.bidirectional else self.H

        # This is a state-space model variant of S4(D) where
        # delta, alpha, beta, gamma, omega directly correspond to
        # the \Delta, A, B, C, D parameters of SSMs
        self.delta = nn.Parameter(torch.Tensor(H, self.N, 1))
        self.alpha = nn.Parameter(torch.Tensor(H, self.N, 1))
        self.beta = nn.Parameter(torch.Tensor(H, self.N))
        self.gamma = nn.Parameter(torch.Tensor(H, self.N))
        self.omega = nn.Parameter(torch.Tensor(self.H))
        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha (dt and A parameters of SSM)
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            # Mega: beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.N)
            if self.N > 1:
                idx = torch.tensor(list(range(1, self.N, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega (C and D parameters of SSM)
            # should be unit variance, as specified in HTTYH
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        p = torch.sigmoid(self.delta)  # (H N 1)
        alpha = torch.sigmoid(self.alpha)
        q = 1.0 - p * alpha
        return p, q

    def _compute_kernel(self, L: int):
        self._kernel = None
        # Materialize parameters - analog of SSM discretization
        p, q = self._calc_coeffs()  # (H N 1)

        vander = torch.log(q) * torch.arange(L).to(p).view(1, 1, L)  # (H N L)
        kernel = p[..., 0] * self.beta * self.gamma * self.scale
        return torch.einsum('dn,dnl->dl', kernel, torch.exp(vander))  # (H L)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, L: int):
        L = L if self.l_max is None else min(self.l_max, L)
        if self.training:
            return self._compute_kernel(L)
        else:
            if self._kernel is None or self._kernel.size(-1) < L:
                self._kernel = self._compute_kernel(L)
            return self._kernel[..., :L]

    def forward(
        self,
        u,
        state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, None]:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        B, L, H = u.size()
        assert H == self.H

        u = u.transpose(-1, -2)  # (B H L)
        if padding_mask is not None:
            u = u * (1.0 - padding_mask.unsqueeze(1).type_as(u))

        # assert not self.bidirectional or state is None, 'Bidirectional EMA does not support incremental state'
        if state is not None:
            raise NotImplementedError(
                "MultiHeadEMA module does not support state passing in this repository."
                "Use S4D for more functionality such as state passing."
            )
        else:
            k = self.kernel(L)  # (H L)
            l_fft = L
            s = 0
            l_kernel = k.size(1)
            assert l_kernel == L
            u_ = u
            if self.bidirectional:
                # This is twice as inefficient as it could be
                # See S4 FFT conv bidirectional implementation for improvement
                k1, k2 = torch.split(k, [self.H, self.H], dim=0)
                k = F.pad(k1, (l_kernel - 1, 0)) + F.pad(k2.flip(-1), (0, l_kernel - 1))  # (H 2*L-1)
                u_ = F.pad(u, (l_kernel - 1, 0))
                l_fft = l_fft + l_kernel - 1
                s = 2 * l_kernel - 2

            k_f = torch.fft.rfft(k.float(), n=2 * l_fft)
            u_f = torch.fft.rfft(u_.float(), n=2 * l_fft)
            y = torch.fft.irfft(u_f * k_f, n=2 * l_fft)[..., s:s + L]  # (B H L)
            y = y.type_as(u)
            y = y + u * self.omega.unsqueeze(-1)  # (B H L)
            y = y.transpose(-1, -2)

        return y, None  # empty state

    def extra_repr(self) -> str:
        return 'edim={}, N={}, bidirectional={}, trunction={}'.format(self.H, self.N, self.bidirectional, self.l_max)

"""
Relative positional bias modules.

From https://github.com/facebookresearch/mega/blob/ea355255149d38ffe16bf2c176d47c3864e8b05a/fairseq/modules/relative_positional_bias.py
"""
class SimpleRelativePositionalBias(nn.Module):

    def __init__(self, max_positions):
        super().__init__()
        self.max_positions = max_positions
        self.rel_pos_bias = nn.Parameter(torch.Tensor(2 * max_positions - 1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.rel_pos_bias, mean=0.0, std=std)

    def forward(self, L):
        if L > self.max_positions:
            raise ValueError('Sequence length {} going beyond max length {}'.format(L, self.max_positions))

        # L * 2 -1
        b = self.rel_pos_bias[(self.max_positions - L):(self.max_positions + L - 1)]
        # L * 3 - 1
        t = F.pad(b, (0, L))
        # (L * 3 - 1) * L
        t = torch.tile(t, (L,))
        t = t[:-L]
        # L x (3 * L - 2)
        t = t.view(L, 3 * L - 2)
        r = (2 * L - 1) // 2
        start = r
        end = t.size(1) - r
        t = t[:, start:end]
        return t

    def extra_repr(self) -> str:
        return 'max positions={}'.format(self.max_positions)


class RotaryRelativePositionalBias(nn.Module):
    def __init__(self, d_model, max_positions):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.max_positions = max_positions
        self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(max_positions, d_model)
        self.alpha = nn.Parameter(torch.Tensor(1, d_model))
        self.beta = nn.Parameter(torch.Tensor(1, d_model))
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.alpha, mean=0.0, std=std)
        nn.init.normal_(self.beta, mean=0.0, std=std)

    @staticmethod
    def get_sinusoid_embeddings(max_positions: int, embedding_dim: int):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(max_positions, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        return torch.sin(emb), torch.cos(emb)

    def rotary(self, x):
        n, d = x.size()
        x1, x2 = torch.chunk(x, 2, dim=-1)
        if self.sine is None or n > self.sine.size(0):
            self.sine, self.cosine = RotaryRelativePositionalBias.get_sinusoid_embeddings(n, d)
            self.max_positions = n
        self.sine = self.sine.to(self._float_tensor)
        self.cosine = self.cosine.to(self._float_tensor)

        sin = self.sine[:n]
        cos = self.cosine[:n]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=1)

    def forward(self, L):
        a = self.rotary(self.alpha.expand(L, self.d_model))
        b = self.rotary(self.beta.expand(L, self.d_model))
        t = torch.einsum('mk,nk->mn', a, b)
        return t

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}'.format(self.d_model, self.max_positions)
