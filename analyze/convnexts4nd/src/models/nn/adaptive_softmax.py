# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adaptive softmax layer for language modeling (e.g. WikiText-103)."""

from typing import List, Optional
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.models.nn.utils as U


class OptionalParameterList(nn.ParameterList):
    def extra_repr(self):
        child_lines = []
        for k, p in self._parameters.items():
            if p is not None:
                size_str = 'x'.join(str(size) for size in p.size())
                device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
                parastr = 'Parameter containing: [{} of size {}{}]'.format(
                    torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 tie_projs=None, out_layers_weights=None, out_projs=None,
                 keep_order=False,
                 bias_scale=0.0,
                 dropout=0.0,
                 ):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj

        self.cutoffs = list(cutoffs) + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.div_val = div_val

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        # bake the first False into the definition, just as [0] is built into the cutoffs
        if tie_projs is None: tie_projs = []
        elif isinstance(tie_projs, bool): tie_projs = [tie_projs] * len(cutoffs)
        else: tie_projs = list(tie_projs)
        tie_projs = [False] + tie_projs
        self.tie_projs = tie_projs

        if self.n_clusters > 0:
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        if not out_layers_weights:
            self.out_layers_weights = nn.ParameterList()
        else:
            self.out_layers_weights = out_layers_weights

        self.out_layers_biases = nn.ParameterList()

        self.shared_out_projs = out_projs
        self.out_projs = OptionalParameterList()

        self.dropout = dropout
        self.drop = nn.Dropout(dropout)

        if div_val == 1:
            if d_proj != d_embed:
                for i in range(len(self.cutoffs)):
                    if tie_projs[i]:
                        self.out_projs.append(None)
                    else:
                        self.out_projs.append(
                            nn.Parameter(torch.zeros(d_proj, d_embed))
                        )
            else:
                self.out_projs.append(None)

            self.out_layers_biases.append(
                nn.Parameter(torch.zeros(n_token))
                )

            if not out_layers_weights:
                self.out_layers_weights.append(
                    nn.Parameter(torch.zeros(n_token, d_embed))
                    )
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)

                if tie_projs[i]:
                    self.out_projs.append(None)
                else:
                    self.out_projs.append(
                        nn.Parameter(torch.zeros(d_proj, d_emb_i))
                    )

                self.out_layers_biases.append(
                    nn.Parameter(torch.zeros(r_idx - l_idx))
                    )
                if not out_layers_weights:
                    self.out_layers_weights.append(
                        nn.Parameter(torch.zeros(r_idx - l_idx, d_emb_i))
                        )
        for bias in self.out_layers_biases:
            bound = bias_scale * d_proj ** -.5
            nn.init.uniform_(bias, -bound, bound)


        self.keep_order = keep_order

    def _compute_logit(self, hidden, weight, bias, proj):
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)
        else:
            if self.dropout > 0.0:
                logit = hidden @ proj
                logit = self.drop(logit)
                logit = logit @ weight.t()
            else:
                logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            if bias is not None:
                logit = logit + bias
        return logit

    def get_out_proj(self, i):
        if self.tie_projs[i]:
            if len(self.shared_out_projs) == 0:
                return None
            elif len(self.shared_out_projs) == 1:
                return self.shared_out_projs[0]
            else:
                return self.shared_out_projs[i]
        else:
            return self.out_projs[i]

    def forward(self, hidden, target, keep_order=False, key_padding_mask=None, *args, **kwargs):
        # [21-09-15 AG]: TODO may need to handle key_padding_mask
        '''
            hidden :: [len*bsz x d_proj]
            target :: [len*bsz]
        '''

        hidden = hidden.reshape(-1, hidden.size(-1))
        target = target.reshape(-1)
        if hidden.size(0) != target.size(0):
            print(hidden.shape, target.shape)
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:
            logit = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))
            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            nll = torch.zeros_like(target, dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs
            for i in range(len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]

                mask_i = (target >= l_idx) & (target < r_idx)
                indices_i = mask_i.nonzero(as_tuple=False).squeeze()

                if indices_i.numel() == 0:
                    continue

                target_i = target.index_select(0, indices_i) - l_idx
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i.gather(1, target_i[:, None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    # First term accounts for cluster probabilities
                    logprob_i = head_logprob_i[:, -i] \
                        + tail_logprob_i.gather(1, target_i[:, None]).squeeze(1)

                if self.keep_order or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0) # TODO This should be a bug in the original implementation; it should go into the continue case above as well

        return nll.mean() # TODO maybe cases for length or padding_mask

    def compute_logits(self, hidden):
        """Compute full vector of logits

        Adapted from https://github.com/kimiyoung/transformer-xl/issues/88
        """
        hidden = hidden.reshape(-1, hidden.size(-1))

        if self.n_clusters == 0:
            logits = self._compute_logit(hidden, self.out_layers_weights[0],
                                        self.out_layers_biases[0], self.get_out_proj(0))
            return logits
        else:
            # construct weights and biases
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    weight_i = self.out_layers_weights[0][l_idx:r_idx]
                    bias_i = self.out_layers_biases[0][l_idx:r_idx]
                else:
                    weight_i = self.out_layers_weights[i]
                    bias_i = self.out_layers_biases[i]

                if i == 0:
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            head_weight, head_bias, head_proj = weights[0], biases[0], self.get_out_proj(0)

            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            head_logprob = F.log_softmax(head_logit, dim=1)

            out_full_logps = [head_logprob[:, :self.cutoffs[0]]]
            offset = 0
            cutoff_values = [0] + self.cutoffs

            for i in range(1, len(cutoff_values) - 1):
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                head_logprob_i = head_logprob # .index_select(0, indices_i)

                if i == 0:
                    logprob_i = head_logprob_i
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.get_out_proj(i)

                    hidden_i = hidden # .index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)
                    logprob_i = head_logprob_i[:, -i].view(-1, 1) + tail_logprob_i

                offset += logprob_i.size(0)
                out_full_logps.append(logprob_i)
            out_full_logps = torch.cat(out_full_logps, dim = 1)
            # print(torch.sum(out_full_ps), out_full_ps.shape)
            return out_full_logps


class AdaptiveEmbedding(nn.Module):
    """ Copy of transformers.AdaptiveEmbedding that works with fp16 by replacing the index_put_ operation

    Initialization has been fixed for the case when d_proj = d_embed
    """
    def __init__(self, n_token, d_embed, d_proj, cutoffs : List[int], div_val=1, init_scale=1.0, sample_softmax=False, dropout=0.0):
        super().__init__()

        self.n_token = n_token
        self.d_embed = d_embed

        self.cutoffs = list(cutoffs) + [n_token]
        self.div_val = div_val
        self.d_proj = d_proj
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.emb_scale = d_proj ** 0.5

        self.cutoff_ends = [0] + self.cutoffs

        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            _init_embed(self.emb_layers[-1].weight, d_embed, init_scale)
            # torch.nn.init.normal_(self.emb_layers[-1].weight, mean=0, std=init_scale * d_embed ** -.5)
            if d_proj != d_embed: # TODO
                # self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
                # torch.nn.init.normal_(self.emb_projs[-1], mean=0, std=init_scale * 1./self.emb_scale)
                _init_proj(self.emb_projs[-1], d_proj, init_scale)
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                # torch.nn.init.normal_(self.emb_layers[-1].weight, mean=0, std=init_scale * d_emb_i ** -.5)
                _init_embed(self.emb_layers[-1].weight, d_emb_i, init_scale)
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))
                # torch.nn.init.normal_(self.emb_projs[-1], mean=0, std=init_scale * 1./self.emb_scale)
                _init_proj(self.emb_projs[-1], d_proj, init_scale)

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            embed = self.drop(embed)
            if self.d_proj != self.d_embed:
                embed = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.reshape(-1)

            # Changes from original impl
            # emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            embeddings = []
            indices = torch.zeros_like(inp_flat) # empty should work as long as cutoffs[-1] > max token
            _total_tokens = 0

            # emb_flat = inp.new_zeros(inp_flat.size(0), self.d_proj)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze(-1) # shape (_tokens,)

                _tokens = indices_i.numel()
                if _tokens == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = self.drop(emb_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                # Changes
                embeddings.append(emb_i)
                indices.index_put_(
                    (indices_i,),
                    torch.arange(_tokens, device=inp.device) + _total_tokens
                )
                _total_tokens += _tokens

                # emb_flat.index_copy_(0, indices_i, emb_i)
            embeddings = torch.cat(embeddings, dim=0)
            emb_flat = embeddings[indices]

            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)

        embed.mul_(self.emb_scale)
        # embed.div_(self.emb_scale)

        return embed


def _init_weight(weight, d : int, init_scale : Optional[float], default=None):
    assert init_scale or default
    if init_scale is None:
        std = default
    else:
        std = init_scale * (d ** -0.5)
    nn.init.normal_(weight, mean=0, std=std)

_init_embed = functools.partial(_init_weight, default=0.02)
_init_proj = functools.partial(_init_weight, default=0.01)
