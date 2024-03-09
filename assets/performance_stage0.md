# Main Results correponding to [arXiv 2401.10166](https://arxiv.org/abs/2401.10166)

### **Classification on ImageNet-1K**

| name | pretrain | resolution |acc@1 | #params | FLOPs | configs/logs/ckpts | best epoch | use ema |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeiT-S | ImageNet-1K | 224x224 | 79.8 | 22M | 4.6G | -- | -- | -- |
| DeiT-B | ImageNet-1K | 224x224 | 81.8 | 86M | 17.5G | -- | -- | -- |
| DeiT-B | ImageNet-1K | 384x384 | 83.1 | 86M | 55.4G | -- | -- | -- |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 28M | 4.5G | -- | -- | -- |
| Swin-S | ImageNet-1K | 224x224 | 83.2 | 50M | 8.7G | -- | -- | -- |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 88M | 15.4G | -- | -- | -- |
| VMamba-T | ImageNet-1K | 224x224 | 82.2 | 22M | ~~4.5G~~ 5.6G | [config](hclassification/configs/vssm/vssm_tiny_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_e292_woema.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmtiny_dp01_ckpt_epoch_292.pth) | 292 | did'nt add |
| VMamba-S | ImageNet-1K | 224x224 | 83.5 | 44M | ~~9.1G~~ 11.2G | [config](classification/configs/vssm/vssm_small_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_e238_ema.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth) | 238 | true |
| VMamba-B | ImageNet-1K | 224x224 | 83.2 | 75M | ~~15.2G~~ 18.0G | [config](classification/configs/vssm/vssm_base_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp05_e260_woema.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp05_ckpt_epoch_260.pth) | 260 | did'nt add |
| VMamba-B* | ImageNet-1K | 224x224 | 83.7 | 75M | ~~15.2G~~ 18.0G | [config](classification/configs/vssm/vssm_base_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp06_e241_ema.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmbase_dp06_ckpt_epoch_241.pth) | 241 | true |

* *Most backbone models trained without ema, `which do not enhance performance \cite(Swin-Transformer)`. We use ema because our model is still under development.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

* *The checkpoints used in object detection and segmentation is `VMamba-B with droppath 0.5` + `no ema`. `VMamba-B*` represents for `VMamba-B with droppath 0.6 + ema`, the performance of which is `non-ema: 83.3 in epoch 262;  ema: 83.7 in epoch 241`. If you are about to use VMamba-B in downstream tasks, try `VMamba-B*` rather than `VMamba-B`, as it is supposed to perform better.*



### **Object Detection on COCO**
  
| Backbone | #params | FLOPs | Detector | box mAP | mask mAP | configs/logs/ckpts | best epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| Swin-T | 48M | 267G | MaskRCNN@1x | 42.7| 39.3 |-- |-- |
| VMamba-T | 42M | ~~262G~~ 286G | MaskRCNN@1x | 46.5| 42.1 | [config](detection/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmtiny_mask_rcnn_swin_fpn_coco.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmtiny_mask_rcnn_swin_fpn_coco_epoch_12.pth) | 12 |
| Swin-S | 69M | 354G | MaskRCNN@1x | 44.8| 40.9 |-- |-- |
| VMamba-S | 64M | ~~357G~~ 400G | MaskRCNN@1x | 48.2| 43.0 | [config](detection/configs/vssm/mask_rcnn_vssm_fpn_coco_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmsmall_mask_rcnn_swin_fpn_coco.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmsmall_mask_rcnn_swin_fpn_coco_epoch_12.pth) | 12 |
| Swin-B | 107M | 496G | MaskRCNN@1x | 46.9| 42.3 |-- |-- |
| VMamba-B | 96M | ~~482G~~ 540G | MaskRCNN@1x | 48.5| 43.1 | [config](detection/configs/vssm/mask_rcnn_vssm_fpn_coco_base.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmbase_mask_rcnn_swin_fpn_coco.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmbase_mask_rcnn_swin_fpn_coco_epoch_12.pth) | 12 |
| Swin-T | 48M | 267G | MaskRCNN@3x | 46.0| 41.6 |-- |-- |
| VMamba-T | 42M | ~~262G~~ 286G | MaskRCNN@3x | 48.5| 43.2 | [config](detection/configs/vssm/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmtiny_mask_rcnn_swin_fpn_coco_ms_3x.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmtiny_mask_rcnn_swin_fpn_coco_ms_3x_epoch_34.pth) | 34 |
| Swin-S | 69M | 354G | MaskRCNN@3x | 48.2| 43.2 |-- |-- |
| VMamba-S | 64M | ~~357G~~ 400G | MaskRCNN@3x | 49.7| 44.0 | [config](detection/configs/vssm/mask_rcnn_vssm_fpn_coco_small_ms_3x.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmsmall_mask_rcnn_swin_fpn_coco_ms_3x.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240222/vssmsmall_mask_rcnn_swin_fpn_coco_ms_3x_epoch_34.pth) | 34 |

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

### **Semantic Segmentation on ADE20K**

| Backbone | Input|  #params | FLOPs | Segmentor | mIoU(SS) | mIoU(MS) | configs/logs/logs(ms)/ckpts | best iter |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |:---: |
| Swin-T | 512x512 | 60M | 945G | UperNet@160k | 44.4| 45.8| -- | -- |
| VMamba-T| 512x512 | 55M | ~~939G~~ 964G | UperNet@160k | 47.3| 48.3| [config](segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmtiny_upernet_4xb4-160k_ade20k-512x512.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmtiny_upernet_4xb4-160k_ade20k-512x512_iter_160000_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmtiny_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth) | 160000 |
| Swin-S | 512x512 | 81M | 1039G | UperNet@160k | 47.6| 49.5| -- | -- |
| VMamba-S| 512x512 | 76M | ~~1037G~~ 1081G | UperNet@160k | 49.5| 50.5|[config](segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmsmall_upernet_4xb4-160k_ade20k-512x512.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmsmall_upernet_4xb4-160k_ade20k-512x512_iter_160000_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmsmall_upernet_4xb4-160k_ade20k-512x512_iter_160000.pth) | 160000 |
| Swin-B | 512x512 | 121M | 1188G | UperNet@160k | 48.1| 49.7|-- |
| VMamba-B| 512x512 | 110M | ~~1167G~~ 1226G | UperNet@160k | 50.0| 51.3|[config](segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmbase_upernet_4xb4-160k_ade20k-512x512.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmbase_upernet_4xb4-160k_ade20k-512x512_iter_128000_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmbase_upernet_4xb4-160k_ade20k-512x512_iter_128000.pth) | 128000 |
| Swin-S | 640x640 | 81M | 1614G | UperNet@160k | 47.9| 48.8| -- | -- |
| VMamba-S| 640x640 | 76M | ~~1620G~~ 1689G | UperNet@160k | 50.8| 50.8| [config](segmentation/configs/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmsmall_upernet_4xb4-160k_ade20k-640x640.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmsmall_upernet_4xb4-160k_ade20k-640x640_iter_112000_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240223/vssmsmall_upernet_4xb4-160k_ade20k-640x640_iter_112000.pth) | 112000 |

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*
