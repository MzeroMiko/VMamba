import os
from functools import partial
import torch

from .vmamba import VSSM


def build_vssm_model(config, **kwargs):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            # ===================
            posembed=config.MODEL.VSSM.POSEMBED,
            imgsize=config.DATA.IMG_SIZE,
        )
        return model

    return None


def build_model(config, is_pretrain=False):
    model = None
    if model is None:
        model = build_vssm_model(config)
    if model is None:
        from .simvmamba import SimVMamba
        if config.MODEL.TYPE in ["sim_tp16"]:
            model = SimVMamba(patch_size=16, embed_dim=192, depth=12, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_tp8"]:
            model = SimVMamba(patch_size=8, embed_dim=192, depth=12, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_sp16"]:
            model = SimVMamba(patch_size=16, embed_dim=384, depth=12, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_sp8"]:
            model = SimVMamba(patch_size=8, embed_dim=384, depth=12, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_bp16"]:
            model = SimVMamba(patch_size=16, embed_dim=768, depth=12, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_bp8"]:
            model = SimVMamba(patch_size=8, embed_dim=768, depth=12, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_lp16"]:
            model = SimVMamba(patch_size=16, embed_dim=1024, depth=24, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
        elif config.MODEL.TYPE in ["sim_lp8"]:
            model = SimVMamba(patch_size=8, embed_dim=1024, depth=24, num_heads="auto", mlp_ratio=4.0, attn_dstate=1, channel_first=False)
    return model




