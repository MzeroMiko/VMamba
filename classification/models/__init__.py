 

from .vmamba.vss import VSS, vss_tiny, vss_small, vss_base
from .vmamba.vmamba import VSSM
from .vmamba.vssmd import VSSMD


# from .build import build_model as build_model_swin
    
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
        return model

    if model_type in ["vssmd"]:
        model = VSSMD(
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
        return model

    if model_type in ["vss"]:
        model = VSS(
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
        return model
    
    return None
    # return build_model_swin(config, is_pretrain)


