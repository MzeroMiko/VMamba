 

from .vmamba.vss import VSS, vss_tiny, vss_small, vss_base
from .vmamba.vmamba import VSSM
from .vmamba.vssmd import VSSMD


from .build import build_model as build_model_swin
    
def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    if model_type in ["vssm"]:
        model = VSSM(
            in_chans=config.MODEL.SWIN.IN_CHANS, 
            patch_size=config.MODEL.SWIN.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.SWIN.DEPTHS, 
            dims=config.MODEL.SWIN.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
        return model

    if model_type in ["vssmd"]:
        model = VSSMD(
            in_chans=config.MODEL.SWIN.IN_CHANS, 
            patch_size=config.MODEL.SWIN.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.SWIN.DEPTHS, 
            dims=config.MODEL.SWIN.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
        return model

    if model_type in ["vss"]:
        model = VSS(
            in_chans=config.MODEL.SWIN.IN_CHANS, 
            patch_size=config.MODEL.SWIN.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.SWIN.DEPTHS, 
            dims=config.MODEL.SWIN.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE
        )
        return model
    
    return build_model_swin(config, is_pretrain)


# python -m torch.distributed.launch --nproc_per_node 8 --master_port 12341  main_ema.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path /dataset/ImageNet2012 --batch-size324 --opts MODEL.SWIN.DEPTHS [2,2,9,2]  TRAIN.ACCUMULATION_STEPS 1 MODEL.TYPE vmamba2dp MODEL.NAME vmamba2dp_tiny



