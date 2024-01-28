import os
import time
import json
import random
import argparse
import datetime
import copy
from typing import Callable
from functools import partial
from collections import OrderedDict

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

from fvcore.nn import FlopCountAnalysis, flop_count

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
build_mmpretrain_models: Callable = build.build_mmpretrain_models
build_vssm_models: Callable = build.build_vssm_models_
selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit

class Logger():
    def info(self, *args, **kwargs):
        if dist.get_rank() == 0:
            print(*args, **kwargs, flush=True)
logger = Logger()


# copied from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# WARNING!!!  acc score would be inaccurate if num_procs > 1, as sampler always pads the dataset
# copied from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


# based on https://github.com/microsoft/Swin-Transformer/blob/main/config.py
# based on https://github.com/microsoft/Swin-Transformer/blob/data/build.py
def build_loader_val(
    root="/dataset/ImageNet2012", 
    shuffle=False, 
    crop=True, 
    img_size=224,
    batch_size=128,
    data_len=-1,
    sequential = False, # if use ddp, should be false
    num_workers = 0,
):
    prefix = "val"

    # transforms.Resize(224) NOT EQUAL TO transforms.Resize((224, 224)) !!!!!!!!!
    def build_transform(resize_im=True, crop=True, img_size=224, interp=InterpolationMode.BICUBIC):
        class Args():
            class TEST():
                CROP = crop
            class DATA():
                IMG_SIZE = img_size
                INTERPOLATION = interp
        config = Args()
        _pil_interp=lambda x: x

        t = []
        if resize_im:
            if config.TEST.CROP:
                size = int((256 / 224) * config.DATA.IMG_SIZE)
                t.append(
                    transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                    # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
            else:
                t.append(
                    transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                    interpolation=_pil_interp(config.DATA.INTERPOLATION))
                )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    transform = build_transform(resize_im=True, crop=crop, img_size=img_size, interp=InterpolationMode.BICUBIC)

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
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return data_loader_val


def _validate(
    model: nn.Module = None, 
    freq=10, 
    amp=True, 
    img_size=224, 
    batch_size=128, 
    data_path="/dataset/ImageNet2012",
):
    class Args():
        ...
    config = Args()
    config.AMP_ENABLE = amp
    config.PRINT_FREQ = freq

    model.cuda().eval()
    model = torch.nn.parallel.DistributedDataParallel(model)
    if isinstance(img_size, tuple or list):
        img_size = img_size[0]
    data_loader = build_loader_val(root=data_path, crop=True, img_size=img_size, batch_size=batch_size)
    logger.info(f"starting loop: img_size {img_size}; len(dataset) {len(data_loader.dataset)}")
    validate(config, data_loader=data_loader, model=model)


def build_models(**kwargs):
    model = None
    if model is None:
        model = build_mmpretrain_models(**kwargs)
    if model is None:
        model = build_vssm_models(**kwargs)
    return model


NAMES = dict(
    tiny=dict(
        vssm="vssm_tiny",
        swin="swin_tiny",
        convnext="convnext_tiny",
        deit="deit_small",
        resnet="resnet50",
    ),
    small=dict(
        vssm="vssm_small",
        swin="swin_small",
        convnext="convnext_small",
        resnet="resnet101",
    ),
    base=dict(
        vssm="vssm_base",
        swin="swin_base",
        convnext="convnext_base",
        deit="deit_base",
        replknet="replknet_base",
    ),
)

# swin do not support size < 224
# replknet when size > 1024: https://github.com/pytorch/pytorch/issues/80020
def main_scale(model="replknet", standard="tiny"):
    kwargs = dict(ckpt=True, only_backbone=False, with_norm=True)
    init_model = partial(build_models, cfg=NAMES[standard][model], **kwargs)
    print(f"using init_{model} ======================== ", flush=True)
    if True:
        _validate(init_model(shape=224).cuda().eval(), img_size=(224, 224), batch_size=128) #128 Mem 6446MB
        if model not in ["swin"]:
            _validate(init_model(shape=64).cuda().eval(), img_size=(64, 64), batch_size=128) #128 Mem 6446MB
        if model not in ["swin"]:
            _validate(init_model(shape=112).cuda().eval(), img_size=(112, 112), batch_size=128) #128 Mem 6446MB
        _validate(init_model(shape=384).cuda().eval(), img_size=(384, 384), batch_size=128)
        _validate(init_model(shape=512).cuda().eval(), img_size=(512, 512), batch_size=128)
        _validate(init_model(shape=640).cuda().eval(), img_size=(640, 640), batch_size=128)
        _validate(init_model(shape=768).cuda().eval(), img_size=(768, 768), batch_size=96)
        _validate(init_model(shape=1024).cuda().eval(), img_size=(1024, 1024), batch_size=(52 if model in ["deit"] else 72))
        _validate(init_model(shape=1120).cuda().eval(), img_size=(1120, 1120), batch_size=(48 if model in ["deit"] else 60))


def main_flops():
    def fvcore_flop_count(model: nn.Module, inputs=None, input_shape=(3, 224, 224)):
        from fvcore.nn.parameter_count import parameter_count
        from fvcore.nn.flop_count import flop_count
        model.eval()
        if inputs is None:
            inputs = (torch.randn((1, *input_shape)).to(next(model.parameters()).device),)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(
            model=model, 
            inputs=inputs, 
            supported_ops={
                "aten::silu": None, # as relu is in _IGNORED_OPS
                "aten::neg": None, # as relu is in _IGNORED_OPS
                "aten::exp": None, # as relu is in _IGNORED_OPS
                "aten::flip": None, # as permute is in _IGNORED_OPS
                "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
            }
        )
        flops = sum(Gflops.values())
        return params, flops

    kwargs = dict(ckpt=True, only_backbone=True, with_norm=True)
    for standard in ["tiny", "small", "base"]:
        print(f"============ {standard} ==============")
        for model in ["vssm", "swin", "convnext", "deit", "resnet"]:
            try:
                cfg=NAMES[standard][model]
            except:
                continue
            init_model = partial(build_models, cfg=NAMES[standard][model], **kwargs)
            for shape in [64, 112, 224, 384, 512, 640, 768, 1024, 1120, 1280]:
                try:
                    params, flops = fvcore_flop_count(init_model(shape=shape), input_shape=(3, shape, shape))
                    print(f"======== model {model} img_size {shape} params {params} flops {flops}", flush=True)
                except:
                    pass


def main(model="replknet", action="tiny"):
    if action in ["tiny", "small", "base"]:
        main_scale(model, action)
    elif action in ['flops']:
        main_flops()


def run_code_dist_one(func: Callable):
    if torch.cuda.device_count() > 1:
        print("WARNING!!!  acc score would be inaccurate if num_procs > 1, as sampler always pads the dataset")
        exit()
        dist.init_process_group(backend='nccl', init_method='env://', world_size=-1, rank=-1)
    else:
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = "61234"
        while True:
            try:
                dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
                break
            except Exception as e:
                print(e, flush=True)
                os.environ['MASTER_PORT'] = f"{int(os.environ['MASTER_PORT']) - 1}"

    torch.cuda.set_device(dist.get_rank())
    dist.barrier()
    func()


if __name__ == "__main__":
    run_code_dist_one(partial(main, os.environ['SCALENET'], os.environ['ACTION']))

