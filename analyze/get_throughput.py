import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# copied from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


# based on https://github.com/microsoft/Swin-Transformer/blob/main/config.py
# based on https://github.com/microsoft/Swin-Transformer/blob/data/build.py
def build_loader_val(
    root="/dataset/ImageNet2012", 
    shuffle=False, 
    crop=True, 
    img_size=(224, 224),
    batch_size=128,
    data_len=-1,
):
    prefix = "val"
    sequential = False # if use ddp, should be false

    t = []
    if crop:
        # to maintain same ratio w.r.t. 224 images
        size = (int((256 / 224) * img_size[0]), int((256 / 224) * img_size[1]))
        t.append(transforms.Resize(size, interpolation=InterpolationMode.BICUBIC))
        t.append(transforms.CenterCrop(img_size))
    else:
        t.append(transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    transform = transforms.Compose(t)

    root = os.path.join(root, prefix)
    dataset_val = datasets.ImageFolder(root, transform=transform)
    
    if data_len is not None and data_len > 0:
        dataset_val.samples = dataset_val.samples[:data_len]

    if sequential:
        sampler_val = SequentialSampler(dataset_val)
    else:
        sampler_val = DistributedSampler(dataset_val, shuffle=shuffle)

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    return data_loader_val


if __name__ == "__main__":
    ...
