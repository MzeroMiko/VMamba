import os
import sys
import torch
import torch.nn as nn

from utils import FLOPs, BuildModels, import_abspy
mmengine_flop_count = FLOPs.mmengine_flop_count
fvcore_flop_count = FLOPs.fvcore_flop_count
mmseg_flops = FLOPs.mmseg_flops
mmdet_flops = FLOPs.mmdet_flops

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
VSSM: nn.Module = build.vmamba.VSSM
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM


def mmdet_mmseg_vssm():
    from mmengine.model import BaseModule
    from mmdet.registry import MODELS as MODELS_MMDET
    from mmseg.registry import MODELS as MODELS_MMSEG

    @MODELS_MMSEG.register_module()
    @MODELS_MMDET.register_module()
    class MM_VSSM(BaseModule, Backbone_VSSM):
        def __init__(self, *args, **kwargs):
            BaseModule.__init__(self)
            Backbone_VSSM.__init__(self, *args, **kwargs)


if __name__ == '__main__':
    # FLOPs.fvcore_flop_count(BuildModels.build_xcit(scale="tiny").cuda())
    # FLOPs.fvcore_flop_count(BuildModels.build_xcit(scale="small").cuda())
    # FLOPs.fvcore_flop_count(BuildModels.build_xcit(scale="base").cuda())
    if True:
        segpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../segmentation/configs")
        detpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../detection/configs")
        mmdet_mmseg_vssm()
        if False:
            FLOPs.mmseg_flops(config=f"{segpath}/upernet/upernet_r50_4xb4-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  952.616667136 Params:  66516108
            FLOPs.mmseg_flops(config=f"{segpath}/upernet/upernet_r101_4xb4-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  1030.4084234239997 Params:  85508236
            FLOPs.mmseg_flops(config=f"{segpath}/vit/vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  1216.821829632 Params:  57994796
            FLOPs.mmseg_flops(config=f"{segpath}/vit/vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  2006.545496064 Params:  144172844
            FLOPs.mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py", input_shape=(3, 512, 2048)) # GFlops:  939.4933174400002 Params:  54546956
            FLOPs.mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py", input_shape=(3, 512, 2048)) # GFlops:  1036.6845167359998 Params:  76070924
            FLOPs.mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py", input_shape=(3, 512, 2048)) # GFlops:  1166.887735664 Params:  109765548
            FLOPs.mmseg_flops(config=f"{segpath}/vssm/upernet_swin_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1614.082896384 Params:  81259766
            FLOPs.mmseg_flops(config=f"{segpath}/vssm/upernet_convnext_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1606.538496 Params:  81877196
            FLOPs.mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1619.8110944 Params:  76070924
    
        if False:
            FLOPs.mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_vssm_fpn_coco_tiny.py") # 42.4M 262093532640.0
            FLOPs.mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_vssm_fpn_coco_small.py") # 63.924M 357006236640.0
            FLOPs.mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_vssm_fpn_coco_base.py") # 95.628M 482127568640.0
            FLOPs.mmdet_flops(config=f"{detpath}/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py") # 44.396M 260152304640.0
            FLOPs.mmdet_flops(config=f"{detpath}/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py") # 63.388M 336434160640.0

        if True:
            FLOPs.mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny1.py", input_shape=(3, 512, 2048)) # GFlops:  947.779848192 Params:  62359340
            FLOPs.mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py", input_shape=(3, 512, 2048)) # GFlops:  948.7801896960001 Params:  61902572
            FLOPs.mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py", input_shape=(3, 512, 2048)) # GFlops:  1028.404888464 Params:  81801260
            FLOPs.mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py", input_shape=(3, 512, 2048)) # GFlops:  1170.3442882240001 Params:  122069292
            FLOPs.mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1606.8682596 Params:  81801260
    
        if True:
            FLOPs.mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_tiny1.py") # 50.212M 270186480640.0
            FLOPs.mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py") # 49.755M 271163376640.0
            FLOPs.mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_small.py") # 69.654M 348921708640.0
            FLOPs.mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_base.py") # 0.108G 485496108640.0

    
