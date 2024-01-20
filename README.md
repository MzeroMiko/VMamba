
<div align="center">
<h1>VMamba </h1>
<h3>VMamba: Visual State Space Model</h3>

[Yue Liu](https://github.com/MzeroMiko)<sup>1</sup>,[Yunjie Tian](https://sunsmarterjie.github.io/)<sup>1</sup>,[Yuzhong Zhao](https://scholar.google.com.hk/citations?user=tStQNm4AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Hongtian Yu](https://github.com/yuhongtian17)<sup>1</sup>, [Lingxi Xie](https://scholar.google.com.hk/citations?user=EEMm7hwAAAAJ&hl=zh-CN&oi=ao)<sup>2</sup>, [Yaowei Wang](https://scholar.google.com.hk/citations?user=o_DllmIAAAAJ&hl=zh-CN&oi=ao)<sup>3</sup>, [Qixiang Ye](https://scholar.google.com.hk/citations?user=tjEfgsEAAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>, [Yunfan Liu](https://scholar.google.com.hk/citations?user=YPL33G0AAAAJ&hl=zh-CN&oi=ao)<sup>1</sup>

<sup>1</sup>  University of Chinese Academy of Sciences, <sup>2</sup>  HUAWEI Inc.,  <sup>3</sup> PengCheng Lab.

Paper: ([arXiv 2401.10166](https://arxiv.org/abs/2401.10166))


</div>

## Updates

* **` Jan. 19th, 2024`:** The source code for classification, object detection, and semantic segmentation are provided. The pre-trained models will be released in few days!


## Abstract

Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) stand as the two most popular foundation models for visual representation learning. While
CNNs exhibit remarkable scalability with linear complexity w.r.t. image resolution, ViTs surpass them in fitting capabilities despite contending with quadratic
complexity. A closer inspection reveals that ViTs achieve superior visual modeling performance through the incorporation of global receptive fields and dynamic
weights. This observation motivates us to propose a novel architecture that inherits these components while enhancing computational efficiency. To this end, we draw
inspiration from the recently introduced state space model and propose the Visual State Space Model (VMamba), which achieves linear complexity without sacrificing global receptive fields. To address the encountered direction-sensitive issue, we introduce the Cross-Scan Module (CSM) to traverse the spatial domain and convert any non-causal visual image into order patch sequences. Extensive experimental results substantiate that VMamba not only demonstrates promising capabilities across various visual perception tasks, but also exhibits more pronounced advantages over established benchmarks as the image resolution increases. 

## Overview

* [**VMamba**](https://arxiv.org/abs/2401.10166) serves as a general-purpose backbone for computer vision with linear complexity and shows the advantages of global receptive fields and dynamic weights.

<p align="center">
  <img src="assets/acc_flow_comp.png" alt="accuracy" width="80%">
</p>

* **2D-Selective-Scan of VMamba**

<p align="center">
  <img src="assets/ss2d.png" alt="arch" width="60%">
</p>

* **VMamba has global effective receptive field**

<p align="center">
  <img src="assets/erf_comp.png" alt="erf" width="50%">
</p>


## Main Results

We will release all the pre-trained models/logs in few days!

* **Classification on ImageNet-1K**


| name | pretrain | resolution |acc@1 | #params | FLOPs | checkpoints/logs |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeiT-S | ImageNet-1K | 224x224 | 79.8 | 22M | 4.6G | -- |
| DeiT-B | ImageNet-1K | 224x224 | 81.8 | 86M | 17.5G | -- |
| DeiT-B | ImageNet-1K | 384x384 | 83.1 | 86M | 55.4G | -- |
| Swin-T | ImageNet-1K | 224x224 | 81.2 | 28M | 4.5G | -- |
| Swin-S | ImageNet-1K | 224x224 | 83.2 | 50M | 8.7G | -- |
| Swin-B | ImageNet-1K | 224x224 | 83.5 | 88M | 15.4G | -- |
| VMamba-T | ImageNet-1K | 224x224 | 82.2 | 22M | 4.5G | waiting |
| VMamba-S | ImageNet-1K | 224x224 | 83.5 | 44M | 9.1G | waiting |
| VMamba-B | ImageNet-1K | 224x224 | 84.0 | 75M | 15.2G | waiting |

* **Object Detection on COCO**
  
| Backbone | #params | FLOPs | Detector | box mAP | mask mAP | checkpoints/logs |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | 48M | 267G | MaskRCNN@1x | 42.7| 39.3 |-- |
| VMamba-T | 42M | 262G | MaskRCNN@1x | 46.5| 42.1 |waiting |
| Swin-S | 69M | 354G | MaskRCNN@1x | 44.8| 40.9 |-- |
| VMamba-S | 64M | 357G | MaskRCNN@1x | 48.2| 43.0 |waiting |
| Swin-B | 107M | 496G | MaskRCNN@1x | 46.9| 42.3 |-- |
| VMamba-B | 96M | 482G | MaskRCNN@1x | 48.5| 43.1 |waiting |
| Swin-T | 48M | 267G | MaskRCNN@3x | 46.0| 41.6 |-- |
| VMamba-T | 42M | 262G | MaskRCNN@3x | 48.5| 43.2 |waiting |
| Swin-S | 69M | 354G | MaskRCNN@3x | 48.2| 43.2 |-- |
| VMamba-S | 64M | 357G | MaskRCNN@3x | 49.7| 44.0 |waiting |

* **Semantic Segmentation on ADE20K**

| Backbone | Input|  #params | FLOPs | Segmentor | mIoU | checkpoints/logs |
| :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| Swin-T | 512x512 | 60M | 945G | UperNet@160k | 44.4| -- |
| VMamba-T| 512x512 | 55M | 939G | UperNet@160k | 47.3| waiting |
| Swin-S | 512x512 | 81M | 1039G | UperNet@160k | 47.6| -- |
| VMamba-S| 512x512 | 76M | 1037G | UperNet@160k | 49.5| waiting |
| Swin-B | 512x512 | 121M | 1188G | UperNet@160k | 48.1| -- |
| VMamba-B| 512x512 | 110M | 1167G | UperNet@160k | 50.0| waiting |
| Swin-S | 640x640 | 81M | 1614G | UperNet@160k | 47.9| -- |
| VMamba-S| 640x640 | 76M | 1620G | UperNet@160k | 50.8| waiting |



## Getting Started

* Install required packages:
```bash
conda_env="vmamba"
nvcc -V
conda create -n ${conda_env} --clone base
python -VV
pip -V
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
# We use py110 cu117 torch113
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

* See more details at [modelcard.sh](modelcard.sh).

## Citation

```
@article{liu2024vmamba,
      title={VMamba: Visual State Space Model}, 
      author={Yue Liu and Yunjie Tian and Yuzhong Zhao and Hongtian Yu and Lingxi Xie and Yaowei Wang and Qixiang Ye and Yunfan Liu},
      journal={arXiv preprint arXiv:2401.10166},
      year={2024}
}
```

## Acknowledgment

This project is based on Mamba ([paper](https://arxiv.org/abs/2312.00752), [code](https://github.com/state-spaces/mamba)), Swin-Transformer ([paper](https://arxiv.org/pdf/2103.14030.pdf), [code](https://github.com/microsoft/Swin-Transformer)), ConvNeXt ([paper](https://arxiv.org/abs/2201.03545), [code](https://github.com/facebookresearch/ConvNeXt)), [OpenMMLab](https://github.com/open-mmlab),
and the analyze/get_erf.py is adopted from [replknet](https://github.com/DingXiaoH/RepLKNet-pytorch/tree/main/erf), thanks for their excellent works.

