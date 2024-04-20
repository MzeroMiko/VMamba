# Main Results correponding to [arXiv 2401.10166v2](https://arxiv.org/abs/2401.10166v2)

### **Classification on ImageNet-1K with VMambav2**

| name | pretrain | resolution |acc@1 | #params | FLOPs | configs/logs/ckpts | best epoch | use ema | GPU Mem | time/epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| DeiT-S | ImageNet-1K | 224x224 | 79.8 | 22M | 4.6G | -- | -- | -- | -- | -- |
| DeiT-B | ImageNet-1K | 224x224 | 81.8 | 86M | 17.5G | -- | -- | -- | -- | -- |
| DeiT-B | ImageNet-1K | 384x384 | 83.1 | 86M | 55.4G | -- | -- | -- | -- | -- |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 28M | 4.5G | -- | -- | -- | -- | -- |
| Swin-S | ImageNet-1K | 224x224 | 83.2 | 50M | 8.7G | -- | -- | -- | -- | -- |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 88M | 15.4G | -- | -- | -- | -- | -- |
| VMamba-T(0230) | ImageNet-1K | 224x224 | 82.5 | 30M | 4.8G | [config](../classification/configs/vssm/vmambav2_tiny_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230.txt)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_tiny_0230_ckpt_epoch_262.pth) | 262 | true | 18234M | 8.12min |
| VMamba-S | ImageNet-1K | 224x224 | 83.6 | 50M | 8.7G | [config](../classification/configs/vssm/vmambav2_small_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_small_0229.txt)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_small_0229_ckpt_epoch_222.pth) | 222 | true | 27634M | 11.86min |
| VMamba-B | ImageNet-1K | 224x224 | 83.9 | 89M | 15.4G | [config](../classification/configs/vssm/vmambav2_base_224.yaml)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_base_0229.txt)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240316/vssm_base_0229_ckpt_epoch_237.pth) | 237 | true | 37122M | 15.08min |

* *Models in this subsection is trained from scratch with random or manual initialization.*

* *We use ema because our model is still under development.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

### **Object Detection on COCO with VMambav2**
  
| Backbone | #params | FLOPs | Detector | box mAP | mask mAP | configs/logs/ckpts | best epoch |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| Swin-T | 48M | 267G | MaskRCNN@1x | 42.7| 39.3 |-- |-- |
| VMamba-T | 50M | 270G | MaskRCNN@1x | 47.4| 42.7 | [config](../detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny_epoch_12.pth) | 12 |
| Swin-S | 69M | 354G | MaskRCNN@1x | 44.8| 40.9 |-- |-- |
| VMamba-S | 70M | 384G | MaskRCNN@1x | 48.7| 43.7 | [config](../detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small_epoch_11.pth) | 11 |
| Swin-B | 107M | 496G | MaskRCNN@1x | 46.9| 42.3 |-- |-- |
| VMamba-B* | 108M | 485G | MaskRCNN@1x | 49.2| 43.9 | [config](../detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_base.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_base_epoch_12.pth) | 12 |
| Swin-T | 48M | 267G | MaskRCNN@3x | 46.0| 41.6 |-- |-- |
| VMamba-T | 50M | 270G | MaskRCNN@3x | 48.9| 43.7 | [config](../detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_epoch_36.pth) | 36 |
| Swin-S | 69M | 354G | MaskRCNN@3x | 48.2| 43.2 |-- |-- |
| VMamba-S | 70M | 384G | MaskRCNN@3x | 49.9| 44.2 | [config](../detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_small_ms_3x.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small_ms_3x.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240320/mask_rcnn_vssm_fpn_coco_small_ms_3x_epoch_32.pth) | 32 |

* *Models in this subsection is initialized from the models trained in `classfication`.*

* *The total batch size of VMamba-B in COCO is `8`, which is supposed to be `16` as in other experiments. This is a `mistake`, not feature. We may fix that later.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

### **Semantic Segmentation on ADE20K with VMambav2**

| Backbone | Input|  #params | FLOPs | Segmentor | mIoU(SS) | mIoU(MS) | configs/logs/logs(ms)/ckpts | best iter |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |:---: |
| Swin-T | 512x512 | 60M | 945G | UperNet@160k | 44.4| 45.8| -- | -- |
| VMamba-T| 512x512 | 62M | 948G | UperNet@160k | 48.3| 48.6| [config](../segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_tiny_iter_160000.pth) | 160k |
| Swin-S | 512x512 | 81M | 1039G | UperNet@160k | 47.6| 49.5| -- | -- |
| VMamba-S| 512x512 | 82M | 1028G | UperNet@160k | 50.6| 51.2|[config](../segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_small.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_small_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_small_iter_144000.pth) | 144k |
| Swin-B | 512x512 | 121M | 1188G | UperNet@160k | 48.1| 49.7|-- |
| VMamba-B| 512x512 | 122M | 1170G | UperNet@160k | 51.0| 51.6|[config](../segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py)/[log](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_base.log)/[log(ms)](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_base_tta.log)/[ckpt](https://github.com/MzeroMiko/VMamba/releases/download/%2320240319/upernet_vssm_4xb4-160k_ade20k-512x512_base_iter_160000.pth) | 160k |
<!-- | Swin-S | 640x640 | 81M | 1614G | UperNet@160k | 47.9| 48.8| -- | -- |
| VMamba-S| 640x640 | 82M | 1607G | UperNet@160k | 50.7| 51.2| [config](../segmentation/configs/vssm1/upernet_vssm_4xb4-160k_ade20k-640x640_small.py)/[log]/[log(ms)]/[ckpt] | 160k | -->

* *Models in this subsection is initialized from the models trained in `classfication`.*

* *we now calculate FLOPs with the algrithm @albertgu [provides](https://github.com/state-spaces/mamba/issues/110), which will be bigger than previous calculation (which is based on the `selective_scan_ref` function, and ignores the hardware-aware algrithm).*

