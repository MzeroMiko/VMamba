"""Implement linear attention.

From github.com/HazyResearch/transformers
"""

from functools import partial
from contextlib import contextmanager
import torch
import torch.nn as nn
import hydra
from einops import rearrange
from fast_transformers.feature_maps import elu_feature_map
from fast_transformers.masking import TriangularCausalMask

from models.sequence.base import SequenceModule, TransposedModule
import src.models.nn.utils as U

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# linear attention classes with softmax kernel

# non-causal linear attention
# By default Performer uses eps=0.0 here
def linear_attention(q, k, v, eps=0.0, need_weights=False):
    k_cumsum = k.sum(dim=-2)
    D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + eps)
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    attn = None if not need_weights else torch.einsum('...te,...se,...s->...ts', q, k, D_inv)
    return out, attn


@contextmanager
def null_context():
    yield


# efficient causal linear attention, created by EPFL
def causal_linear_attention(q, k, v, eps=1e-6, need_weights=False):
    from fast_transformers.causal_product import CausalDotProduct
    autocast_enabled = torch.is_autocast_enabled()
    is_half = isinstance(q, torch.cuda.HalfTensor)
    assert not is_half or APEX_AVAILABLE, 'half tensors can only be used if nvidia apex is available'
    cuda_context = null_context if not autocast_enabled else partial(autocast, enabled = False)

    causal_dot_product_fn = amp.float_function(CausalDotProduct.apply) if is_half else CausalDotProduct.apply

    k_cumsum = k.cumsum(dim=-2) + eps
    D_inv = 1. / torch.einsum('...nd,...nd->...n', q, k_cumsum.type_as(q))

    with cuda_context():
        if autocast_enabled:
            q, k, v = map(lambda t: t.float(), (q, k, v))
        out = causal_dot_product_fn(q, k, v)
        if need_weights:
            attn = torch.einsum('...te,...se,...s', q, k, D_inv)
            causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool,
                                                device=k.device), diagonal=1)
            attn.masked_fill_(causal_mask, 0.0)
        else:
            attn = None

    out = torch.einsum('...nd,...n->...nd', out, D_inv)
    return out, None


# inefficient causal linear attention, without cuda code, for reader's reference
# not being used

# Adapted from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
class LinearAttention(nn.Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.
    Given the query, key and value as Q, K, V instead of computing
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    we make use of a feature map function Φ(.) and perform the following
    computation
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    # def __init__(self, query_dims, feature_map_cfg=None, eps=1e-6):
    def __init__(self, d_model, n_heads, feature_map_cfg=None, eps=1e-6, dropout=0.0): # TODO dropout not used
        super().__init__()
        query_dims = d_model // n_heads
        self.n_heads = n_heads
        self.feature_map = (
            hydra.utils.instantiate(feature_map_cfg, query_dims) if feature_map_cfg is not None
            else elu_feature_map(query_dims)
        )
        self.eps = eps

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        # Permute the dimensions to BHTE instead of BTHE
        query = rearrange(query, 'b t (h e) -> b h t e', h=self.n_heads)
        key = rearrange(key, 'b s (h e) -> b h s e', h=self.n_heads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.n_heads)

        # Apply the feature map to the query and key
        self.feature_map.new_feature_map(query.device)
        Q = self.feature_map.forward_queries(query)
        K = self.feature_map.forward_keys(key)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones or is causal
        causal = attn_mask is not None and attn_mask.lower_triangular
        if not (attn_mask is None or attn_mask.all_ones or causal):
            raise RuntimeError(("LinearAttention does not support arbitrary attention masks"))
        if causal:
            assert query.shape[1] == key.shape[1], 'query and key must have the same sequence length'

        if key_padding_mask is not None:
            K.masked_fill_(~rearrange(key_padding_mask.bool_matrix, 'b s -> b 1 s 1'), 0.0)
        attn_fn = causal_linear_attention if causal else linear_attention
        out, attn = attn_fn(Q, K, value, eps=self.eps, need_weights=need_weights)
        out = rearrange(out, 'b h s d -> b s (h d)')
        return out, attn

@TransposedModule
class Performer(SequenceModule):
    # TODO [21-09-29] the MHA class should take options for attention like full, performer, etc. Currently this is essentially duplicated from MultiheadAttention class."""
    def __init__(self, d_model, n_heads, *args, causal=True, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_output = d_model
        self.mha = LinearAttention(d_model, n_heads, *args, **kwargs)
        self.causal = causal

    def forward(self, src, attn_mask=None, key_padding_mask=None, state=None, **kwargs):
        """ state should represent a mask and key padding mask """
        if self.causal and attn_mask is None:
            attn_mask = TriangularCausalMask(src.size(-2), device=src.device)
        # attn_mask, key_padding_mask = state
        # Note that this returns None for the second argument
        y, z = self.mha(src, src, src, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        return y, None

    def step(self, x, state):
        raise NotImplementedError
