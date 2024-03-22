"""End-to-end classification Transformer adapted from PyTorch examples.

The isotropic model backbone should subsume this architecture. See config configs/model/transformer.yaml
"""

import copy
from typing import Optional, Any
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.init import xavier_uniform_
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter


class ClassificationTransformer(Module):

    def __init__(
            self,
            d_input,
            d_output,
            d_model: int = 512,
            nhead: int = 8,
            num_encoder_layers: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "gelu",
            prenorm: bool = False,
            **kwargs,
    ) -> None:
        super().__init__()

        # Input projection to make the number of channels `d_model`
        self.input_proj = torch.nn.Linear(
            in_features=d_input,
            out_features=d_model,
        )

        # Create the TransformerEncoder blocks
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, activation, share_qk=False, prenorm=prenorm
            ),
            num_encoder_layers,
            LayerNorm(d_model)
        )

        # Output projection
        self.output_proj = torch.nn.Linear(
            in_features=d_model,
            out_features=d_output,
        )
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(
            self,
            src: Tensor,
            *args,
            **kwargs
    ) -> Tensor:

        # Encode the input (B, S, C)
        x = self.input_proj(src)
        x = self.encoder.forward(x)
        return self.output_proj(x[:, -1, :])  # uses the encoding of the last "token" to classify

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class Transformer(Module):
    r"""A transformer model. User is able to modify the attributes as needed. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information
    Processing Systems, pages 6000-6010. Users can build the BERT(https://arxiv.org/abs/1810.04805)
    model with corresponding parameters.
    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, relu or gelu (default=relu).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    Examples::
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)
    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    """

    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None, approx: dict = None) -> None:
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                    share_qk=False)
            encoder_norm = LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation,
                                                    share_qk=False)
            decoder_norm = LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Take in and process masked source/target sequences.
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
        Shape:
            - src: :math:`(S, N, E)`.
            - tgt: :math:`(T, N, E)`.
            - src_mask: :math:`(S, S)`.
            - tgt_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero
            positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - output: :math:`(T, N, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        Examples:
            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        """

        if src.size(1) != tgt.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, types: Optional[dict] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.
        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = src
        for mod in self.layers:
            output = mod(output, types=types, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, types: Optional[dict] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, types=types, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
            self,
            d_model,
            nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            share_qk=False,
            prenorm=False,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, share_qk=share_qk)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.prenorm = prenorm

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, types: Optional[dict] = None, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        if self.prenorm:
            # src = self.norm1(src)
            src2 = self.norm1(src)
            src2 = self.self_attn(src2, src2, src2, types=types, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
        else:
            # Old code
            src2 = self.self_attn(src, src, src, types=types, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)

        if self.prenorm:
            src2 = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
            src = src + self.dropout2(src2)
        else:
            # Old code
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)
        return src


class TransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", share_qk=False,
                 approx=None):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def kl(p, q):
    kl_dis = F.kl_div(p, q)
    return kl_dis


def mse(p, q):
    mse_loss = F.mse_loss(p, q)
    return mse_loss


def l1(p, q):
    l1_loss = F.l1_loss(p, q)
    return l1_loss


def smart_sort(x, permutation):
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        permutation.flatten()
    ].view(d1, d2)
    return ret


def sparsify(target, params_reduction):
    target_sparse = target.clone()
    N, target_l, seq_l = target_sparse.shape
    sorted_tensor, indices_tensor = torch.sort(target_sparse, dim=-1, descending=True)
    topk = int(round(seq_l * (1 - params_reduction)))
    mask = torch.zeros_like(target_sparse, dtype=torch.bool).scatter_(-1, indices_tensor[:, :, :topk], 1)
    target_sparse[~mask] = float(
        '-inf')  # To zero out these values, we set their logit to be -inf, so that after softmax they are zero
    return target_sparse, mask.bool()


def low_rank(target, sparsity):
    N, target_l, seq_l = target.shape
    target_lr = target.clone()
    try:
        u, s, v = torch.svd(target_lr)
        topk = int(round(seq_l * (1 - sparsity)))
        # assert torch.dist(target_lr, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1)))<1e-2
        s[:, topk:] = 0
        target_lr = torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1))
        return target_lr, True
    except:  # torch.svd may have convergence issues for GPU and CPU.
        return target_lr, False


def log_stats(approx, target):
    eps = 1e-5
    sparse_l1 = l1(approx, target)
    sparse_kl = kl(torch.log(approx + eps), target + eps)
    sparse_kl_inverse = kl(torch.log(target + eps), approx + eps)
    return torch.cat([sparse_l1.view(1), sparse_kl.view(1), sparse_kl_inverse.view(1)])


def compute_single_distance(target_raw, attn_mask, params_reduction, approx_type, alpha=0.5):
    stats = torch.zeros([1, 3])
    target_raw[target_raw < -1e7] = float('-inf')
    target = F.softmax(target_raw, dim=-1)
    succeed = True
    approx_target = 0

    # sparse
    if approx_type == "sparse":
        target_sparse, mask = sparsify(target_raw, params_reduction)
        if attn_mask is not None:
            target_sparse.masked_fill_(attn_mask, float('-inf'), )
        approx_target = torch.softmax(target_sparse, dim=-1)
        stats = log_stats(approx_target, target)

    # low_rank
    elif approx_type == "low_rank":
        new_sparsity = 1 - (1 - params_reduction) / 2
        target_lr, succeed = low_rank(target, new_sparsity)
        if succeed:
            target_lr[target_lr < 0] = 0.0
            if attn_mask is not None:
                target_lr.masked_fill_(attn_mask, 0.0, )
            approx_target = F.normalize(target_lr, p=1, dim=-1)
            stats = log_stats(approx_target, target)

    # sparse+low_rank
    elif approx_type == "sparse_low_rank":
        target_sparse = target.clone()
        params_sparse = alpha * (1 - params_reduction)
        _, mask = sparsify(target, 1 - params_sparse)
        target_sparse[~mask] = 0.0
        target_sparse_lr = target - target_sparse
        params_lr = (1 - alpha) * (1 - params_reduction) / 2
        target_sparse_lr, succeed = low_rank(target_sparse_lr, 1 - params_lr)
        if succeed:
            target_sparse_lr[target_sparse_lr < 0] = 0.0
            target_sparse_lr += target_sparse
            if attn_mask is not None:
                target_sparse_lr.masked_fill_(attn_mask, 0.0, )
            approx_target = F.normalize(target_sparse_lr, p=1, dim=-1)
            stats = log_stats(approx_target, target)
    else:
        print("Approximation type is not implemented")
    return approx_target, stats, succeed


class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None, share_qk=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.bias = bias
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.q_proj_weight = torch.nn.Linear(embed_dim, embed_dim, bias=self.bias)
        self.k_proj_weight = torch.nn.Linear(embed_dim, self.kdim, bias=self.bias)
        self.v_proj_weight = torch.nn.Linear(embed_dim, self.vdim, bias=self.bias)
        xavier_uniform_(self.q_proj_weight.weight)
        xavier_uniform_(self.k_proj_weight.weight)
        xavier_uniform_(self.v_proj_weight.weight)
        self.out_proj = torch.nn.Linear(embed_dim, self.vdim)

        #         self._reset_parameters()

        if self.bias:
            constant_(self.q_proj_weight.bias, 0.)
            constant_(self.v_proj_weight.bias, 0.)
            constant_(self.out_proj.bias, 0.)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
            xavier_normal_(self.bias_k)
            xavier_normal_(self.bias_v)

        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        if share_qk:
            self.in_proj_container = SharedQK_Proj(self.q_proj_weight, self.v_proj_weight)
        else:
            self.in_proj_container = InProjContainer(self.q_proj_weight, self.k_proj_weight, self.v_proj_weight)
        self.multihead_attention = MultiheadAttentionContainer(num_heads,
                                                               self.in_proj_container,
                                                               ScaledDotProduct(self.dropout),
                                                               self.out_proj)

    def forward(self, query, key, value, types=None, key_padding_mask=None, need_weights=True, attn_mask=None):
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(-1, attn_mask.size(0), attn_mask.size(1))
            attn_mask = attn_mask.bool()
        return self.multihead_attention(query, key, value, types, attn_mask, self.bias_k, self.bias_v)


class MultiheadAttentionContainer(torch.nn.Module):
    def __init__(self, nhead, in_proj_container, attention_layer, out_proj):
        r""" A multi-head attention container
        Args:
            nhead: the number of heads in the multiheadattention model
            in_proj_container: A container of multi-head in-projection linear layers (a.k.a nn.Linear).
            attention_layer: The attention layer.
            out_proj: The multi-head out-projection layer (a.k.a nn.Linear).
        Examples::
            >>> import torch
            >>> embed_dim, num_heads, bsz = 10, 5, 64
            >>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim),
                                                    torch.nn.Linear(embed_dim, embed_dim))
            >>> MHA = MultiheadAttentionContainer(num_heads,
                                                  in_proj_container,
                                                  ScaledDotProduct(),
                                                  torch.nn.Linear(embed_dim, embed_dim))
            >>> query = torch.rand((21, bsz, embed_dim))
            >>> key = value = torch.rand((16, bsz, embed_dim))
            >>> attn_output, attn_weights = MHA(query, key, value)
            >>> print(attn_output.shape)
            >>> torch.Size([21, 64, 10])
        """
        super(MultiheadAttentionContainer, self).__init__()
        self.nhead = nhead
        self.in_proj_container = in_proj_container
        self.attention_layer = attention_layer
        self.out_proj = out_proj
        self.attn_map = 0

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                types: Optional[dict] = None,
                attn_mask: Optional[torch.Tensor] = None,
                bias_k: Optional[torch.Tensor] = None,
                bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            query, key, value (Tensor): map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            attn_mask, bias_k and bias_v (Tensor, optional): keyword arguments passed to the attention layer.
                See the definitions in the attention.
        Shape:
            - Inputs:
            - query: :math:`(L, N, E)`
            - key: :math:`(S, N, E)`
            - value: :math:`(S, N, E)`
            - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.
            - Outputs:
            - attn_output: :math:`(L, N, E)`
            - attn_output_weights: :math:`(N * H, L, S)`
            where where L is the target length, S is the sequence length, H is the number of attention heads,
                N is the batch size, and E is the embedding dimension.
        """
        tgt_len, src_len, bsz, embed_dim = query.size(-3), key.size(-3), query.size(-2), query.size(-1)
        q, k, v = self.in_proj_container(query, key, value)
        assert q.size(-1) % self.nhead == 0, "query's embed_dim must be divisible by the number of heads"
        head_dim = q.size(-1) // self.nhead
        q = q.reshape(tgt_len, bsz * self.nhead, head_dim)

        assert k.size(-1) % self.nhead == 0, "key's embed_dim must be divisible by the number of heads"
        head_dim = k.size(-1) // self.nhead
        k = k.reshape(src_len, bsz * self.nhead, head_dim)

        assert v.size(-1) % self.nhead == 0, "value's embed_dim must be divisible by the number of heads"
        head_dim = v.size(-1) // self.nhead
        v = v.reshape(src_len, bsz * self.nhead, head_dim)

        attn_output, attn_output_weights, self.attn_map = self.attention_layer(q, k, v,
                                                                               types=types, attn_mask=attn_mask,
                                                                               bias_k=bias_k, bias_v=bias_v)
        attn_output = attn_output.reshape(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_output_weights


class ScaledDotProduct(torch.nn.Module):

    def __init__(self, dropout=0.0):
        r"""Processes a projected query and key-value pair to apply
        scaled dot product attention.
        Args:
            dropout (float): probability of dropping an attention weight.
        Examples::
            >>> SDP = torchtext.models.ScaledDotProduct(0.1)
            >>> q = torch.randn(256, 21, 3)
            >>> k = v = torch.randn(256, 21, 3)
            >>> attn_output, attn_weights = SDP(q, k, v)
            >>> print(attn_output.shape, attn_weights.shape)
            torch.Size([256, 21, 3]) torch.Size([256, 21, 21])
        """
        super(ScaledDotProduct, self).__init__()
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                types: Optional[dict] = None,
                attn_mask: Optional[torch.Tensor] = None,
                bias_k: Optional[torch.Tensor] = None,
                bias_v: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Uses a scaled dot product with the projected key-value pair to update
        the projected query.
        Args:
            query (Tensor): Projected query
            key (Tensor): Projected key
            value (Tensor): Projected value
            attn_mask (BoolTensor, optional): 3D mask that prevents attention to certain positions.
            bias_k and bias_v: (Tensor, optional): one more key and value sequence to be added at
                sequence dim (dim=-3). Those are used for incremental decoding. Users should provide
                non-None to both arguments in order to activate them.
        Shape:
            - query: :math:`(L, N * H, E / H)`
            - key: :math:`(S, N * H, E / H)`
            - value: :math:`(S, N * H, E / H)`
            - attn_mask: :math:`(N * H, L, S)`, positions with ``True`` are not allowed to attend
                while ``False`` values will be unchanged.
            - bias_k and bias_v:bias: :math:`(1, N * H, E / H)`
            - Output: :math:`(L, N * H, E / H)`, :math:`(N * H, L, S)`
            where L is the target length, S is the source length, H is the number
            of attention heads, N is the batch size, and E is the embedding dimension.
        """
        if bias_k is not None and bias_v is not None:
            assert key.size(-1) == bias_k.size(-1) and key.size(-2) == bias_k.size(-2) and bias_k.size(-3) == 1, \
                "Shape of bias_k is not supported"
            assert value.size(-1) == bias_v.size(-1) and value.size(-2) == bias_v.size(-2) and bias_v.size(-3) == 1, \
                "Shape of bias_v is not supported"
            key = torch.cat([key, bias_k])
            value = torch.cat([value, bias_v])
            if attn_mask is not None:
                _attn_mask = attn_mask
                attn_mask = torch.nn.functional.pad(_attn_mask, (0, 1))

        tgt_len, head_dim = query.size(-3), query.size(-1)
        assert query.size(-1) == key.size(-1) == value.size(-1), "The feature dim of query, key, value must be equal."
        assert key.size() == value.size(), "Shape of key, value must match"
        src_len = key.size(-3)
        batch_heads = max(query.size(-2), key.size(-2))

        # Scale query
        query, key, value = query.transpose(-2, -3), key.transpose(-2, -3), value.transpose(-2, -3)
        query = query * (float(head_dim) ** -0.5)
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise RuntimeError('attn_mask must be a 3D tensor.')
            if (attn_mask.size(-1) != src_len) or (attn_mask.size(-2) != tgt_len) or \
                    (attn_mask.size(-3) != 1 and attn_mask.size(-3) != batch_heads):
                raise RuntimeError('The size of the attn_mask is not correct.')
            if attn_mask.dtype != torch.bool:
                raise RuntimeError('Only bool tensor is supported for attn_mask')

        # Dot product of q, k
        attn_output_weights = torch.matmul(query, key.transpose(-2, -1))
        if attn_mask is not None:
            attn_output_weights.masked_fill_(attn_mask, -1e8, )
        attn_map = {}
        attn_map['attn'] = attn_output_weights
        attn_map['stat'] = None
        attn_map['succeed'] = None

        # approx attn weights
        if (types is not None) and (not self.training):
            attn_output_weights, attn_map['stat'], attn_map['succeed'] = compute_single_distance \
                (attn_map['attn'], attn_mask, types['params_reduction'],
                 types['approx_type'], alpha=types['alpha'])
        else:
            attn_output_weights = torch.nn.functional.softmax(attn_output_weights, dim=-1)

        attn_output_weights = torch.nn.functional.dropout(attn_output_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_output_weights, value)
        return attn_output.transpose(-2, -3), attn_output_weights, attn_map


class SharedQK_Proj(torch.nn.Module):
    def __init__(self, qk_proj, v_proj):
        super(SharedQK_Proj, self).__init__()
        self.qk_proj = qk_proj
        self.v_proj = qk_proj

    def forward(self, q, k, v):
        return self.qk_proj(q), self.qk_proj(k), self.v_proj(v)


class InProjContainer(torch.nn.Module):
    def __init__(self, query_proj, key_proj, value_proj):
        r"""A in-proj container to process inputs.
        Args:
            query_proj: a proj layer for query.
            key_proj: a proj layer for key.
            value_proj: a proj layer for value.
        """

        super(InProjContainer, self).__init__()
        self.query_proj = query_proj
        self.key_proj = key_proj
        self.value_proj = value_proj

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Projects the input sequences using in-proj layers.
        Args:
            query, key, value (Tensors): sequence to be projected
        Shape:
            - query, key, value: :math:`(S, N, E)`
            - Output: :math:`(S, N, E)`
            where S is the sequence length, N is the batch size, and E is the embedding dimension.
        """
        return self.query_proj(query), self.key_proj(key), self.value_proj(value)


def generate_square_subsequent_mask(nbatch, sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with True.
        Unmasked positions are filled with False.
    Args:
        nbatch: the number of batch size
        sz: the size of square mask
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).repeat(nbatch, 1, 1)
    return mask
