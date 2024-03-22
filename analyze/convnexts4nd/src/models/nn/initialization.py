"""Original from Transformer-XL as a hook for their initialization. Currently not used."""

import torch
from torch import nn

def init_weight(weight, init_cfg):
    if init_cfg.init == 'uniform':
        nn.init.uniform_(weight, -init_cfg.init_range, init_cfg.init_range)
    elif init_cfg.init == 'normal':
        nn.init.normal_(weight, 0.0, init_cfg.init_std)
    elif init_cfg.init == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif init_cfg.init == 'kaiming':
        nn.init.kaiming_normal_(weight, mode='fan_in', nonlinearity='linear')
    else:
        raise NotImplementedError(f"initialization type {init_cfg.init} not supported")

def init_bias(bias, init_cfg):
    if hasattr(init_cfg, 'zero_bias') and init_cfg.zero_bias==False:
        # Keep the original bias init
        pass
    else:
        nn.init.constant_(bias, 0.0)

def weights_init(m, init_cfg):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, init_cfg)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias, init_cfg)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            if hasattr(init_cfg, 'ln') and init_cfg.ln==False:
                pass
            else:
                nn.init.normal_(m.weight, 1.0, init_cfg.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias, init_cfg)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb, init_cfg)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, init_cfg)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, init_cfg)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias, init_cfg)
        if hasattr(m, 'initial_state'):
            init_bias(m.initial_state, init_cfg)

def weights_init_embedding(m, init_cfg):
    classname = m.__class__.__name__
    if classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, init_cfg.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, init_cfg)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, init_cfg)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias, init_cfg)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, init_cfg.proj_init_std)
        if hasattr(m, 'out_layers_weights'):
            for i in range(len(m.out_layers_weights)):
                if m.out_layers_weights[i] is not None:
                    init_weight(m.out_layers_weights[i], init_cfg)
