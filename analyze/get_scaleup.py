import os
import time
import json
import random
import argparse
import datetime
import numpy as np
import copy
from functools import partial

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from collections import OrderedDict

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

# FALSE!!!
# based on https://github.com/microsoft/Swin-Transformer/blob/main/config.py
# based on https://github.com/microsoft/Swin-Transformer/blob/data/build.py
def build_loader_val_v0(
    root="/dataset/ImageNet2012", 
    shuffle=False, 
    crop=True, 
    img_size=(224, 224),
    batch_size=128,
    data_len=-1,
    sequential = False, # if use ddp, should be false
):
    prefix = "val"

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
):
    prefix = "val"

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
        num_workers=0,
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


# ======================================
def build_vssm(ckpt=None, only_backbone=False, with_norm=True, depths=[2, 2, 9, 2], dims=96, **kwargs):
    from vmamba.vmamba import VSSM
    model = VSSM(depths=depths, dims=dims)
        
    if isinstance(ckpt, str) and os.path.exists(ckpt):
        _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))['model']
        print(f"Successfully load ckpt {ckpt}")
        model.load_state_dict(_ckpt)

    if only_backbone:
        def forward_backbone(self: VSSM, x, with_norm=False):
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

            for layer in self.layers:
                x = layer(x)
            if with_norm:
                B, H, W, C = x.shape
                x = torch.flatten(x, 1, 2) # B H W C -> B L C
                x = self.norm(x)  # B L C
                x = x.view(B, H, W, C)
            x = x.permute(0, 3, 1, 2).contiguous() 
            return x
        
        model.forward = partial(forward_backbone, model, with_norm=with_norm)

        if not with_norm:
            del model.norm
        del model.avgpool
        del model.head
        
    model.cuda().eval()
        
    return model


def build_mmpretrain_models(cfg="swin_tiny", ckpt=True, only_backbone=False, with_norm=True, **kwargs):
    from mmengine.runner import CheckpointLoader
    from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
    from mmengine.config import Config
    config_root = os.path.join(os.path.dirname(__file__), "./mmpretrain_configs/configs/") 
    
    CFGS = dict(
        swin_tiny=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-tiny_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth",
        ),
        convnext_tiny=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-tiny_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-tiny_32xb128_in1k_20221207-998cf3e9.pth",
        ),
        deit_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./deit/deit-small_4xb256_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth",
        ),
        resnet50=dict(
            model=Config.fromfile(os.path.join(config_root, "./resnet/resnet50_8xb32_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth",
        ),
        # ================================
        swin_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-small_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_small_224_b16x64_300e_imagenet_20210615_110219-7f9d988b.pth",
        ),
        convnext_small=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-small_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth",
        ),
        deit_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./deit/deit-base_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth",
        ),
        resnet101=dict(
            model=Config.fromfile(os.path.join(config_root, "./resnet/resnet101_8xb32_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth",
        ),
        # ================================
        swin_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./swin_transformer/swin-base_16xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_base_224_b16x64_300e_imagenet_20210616_190742-93230b0d.pth",
        ),
        convnext_base=dict(
            model=Config.fromfile(os.path.join(config_root, "./convnext/convnext-base_32xb128_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/convnext/convnext-base_32xb128_in1k_20221207-fbdb5eb9.pth",
        ),
        replknet_base=dict(
            # comment this "from mmpretrain.models import build_classifier" in __base__/models/replknet...
            model=Config.fromfile(os.path.join(config_root, "./replknet/replknet-31B_32xb64_in1k.py")).to_dict()['model'], 
            ckpt="https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth",
        ),
    )

    model: ImageClassifier = build_classifier(CFGS[cfg]['model'])
    if ckpt:
        model.load_state_dict(CheckpointLoader.load_checkpoint(CFGS[cfg]['ckpt'])['state_dict'])

    if only_backbone:
        if isinstance(model.backbone, ConvNeXt):
            model.backbone.gap_before_final_norm = False
        if isinstance(model.backbone, VisionTransformer):
            model.backbone.out_type = 'featmap'

        def forward_backbone(self: ImageClassifier, x):
            x = self.backbone(x)[-1]
            return x
        if not with_norm:
            setattr(model, f"norm{model.backbone.out_indices[-1]}", lambda x: x)
        model.forward = partial(forward_backbone, model)

    model.cuda().eval()
    
    return model


def main_tiny(model="replknet"):
    init_vssm = partial(build_vssm, ckpt=os.path.join(os.path.dirname(__file__), "../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth"))
    init_swin = partial(build_mmpretrain_models, cfg="swin_tiny", ckpt=True, only_backbone=False, with_norm=True)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=True, only_backbone=False, with_norm=True)
    init_deit = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=False, with_norm=True)
    init_resnet = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=False, with_norm=True)

    # replknet when size > 1024
    # https://github.com/pytorch/pytorch/issues/80020

    print(f"using init_{model} ======================== ", flush=True)
    init_model = eval(f"init_{model}")
    
    if True:
        _validate(init_model(shape=224), img_size=(224, 224), batch_size=128) #128 Mem 6446MB
        try:
            _validate(init_model(shape=64), img_size=(64, 64), batch_size=128) #128 Mem 6446MB
        except Exception as e:
            print(e)
        try:
            _validate(init_model(shape=112), img_size=(112, 112), batch_size=128) #128 Mem 6446MB
        except Exception as e:
            print(e)
        _validate(init_model(shape=384), img_size=(384, 384), batch_size=128)
        _validate(init_model(shape=512), img_size=(512, 512), batch_size=128)
        _validate(init_model(shape=640), img_size=(640, 640), batch_size=128)
        _validate(init_model(shape=768), img_size=(768, 768), batch_size=96)
    if True:
        _validate(init_model(shape=1024), img_size=(1024, 1024), batch_size=(52 if model in ["deit"] else 72))
        _validate(init_model(shape=1120), img_size=(1120, 1120), batch_size=(48 if model in ["deit"] else 60))
        # _validate(init_model(shape=1280), img_size=(1280, 1280), batch_size=32)

def main_small(model="replknet"):
    init_vssm = partial(build_vssm, ckpt=os.path.join(os.path.dirname(__file__), "../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth"), depths=[2,2,27,2])
    init_swin = partial(build_mmpretrain_models, cfg="swin_small", ckpt=True, only_backbone=False, with_norm=True)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_small", ckpt=True, only_backbone=False, with_norm=True)
    init_resnet = partial(build_mmpretrain_models, cfg="resnet101", ckpt=True, only_backbone=False, with_norm=True)

    # replknet when size > 1024
    # https://github.com/pytorch/pytorch/issues/80020

    print(f"using init_{model} ======================== ", flush=True)
    init_model = eval(f"init_{model}")
    
    if True:
        _validate(init_model(shape=224), img_size=(224, 224), batch_size=128) #128 Mem 6446MB
        try:
            _validate(init_model(shape=64), img_size=(64, 64), batch_size=128) #128 Mem 6446MB
        except Exception as e:
            print(e)
        try:
            _validate(init_model(shape=112), img_size=(112, 112), batch_size=128) #128 Mem 6446MB
        except Exception as e:
            print(e)
        _validate(init_model(shape=384), img_size=(384, 384), batch_size=128)
        _validate(init_model(shape=512), img_size=(512, 512), batch_size=128)
        _validate(init_model(shape=640), img_size=(640, 640), batch_size=128)
        _validate(init_model(shape=768), img_size=(768, 768), batch_size=96)
    if True:
        _validate(init_model(shape=1024), img_size=(1024, 1024), batch_size=(36 if model in ["deit"] else 72))
        _validate(init_model(shape=1120), img_size=(1120, 1120), batch_size=(24 if model in ["deit"] else 60))
        # _validate(init_model(shape=1280), img_size=(1280, 1280), batch_size=32)

def main_base(model="replknet"):
    init_vssm = partial(build_vssm, ckpt=os.path.join(os.path.dirname(__file__), "../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth"), depths=[2,2,27,2], dims=128)
    init_swin = partial(build_mmpretrain_models, cfg="swin_base", ckpt=True, only_backbone=False, with_norm=True)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_base", ckpt=True, only_backbone=False, with_norm=True)
    init_deit = partial(build_mmpretrain_models, cfg="deit_base", ckpt=True, only_backbone=False, with_norm=True)
    init_replknet = partial(build_mmpretrain_models, cfg="replknet_base", ckpt=True, only_backbone=False, with_norm=True)

    # replknet when size > 1024
    # https://github.com/pytorch/pytorch/issues/80020

    print(f"using init_{model} ======================== ", flush=True)
    init_model = eval(f"init_{model}")
    
    if True:
        _validate(init_model(shape=224), img_size=(224, 224), batch_size=128) #128 Mem 6446MB
        try:
            _validate(init_model(shape=64), img_size=(64, 64), batch_size=128) #128 Mem 6446MB
        except Exception as e:
            print(e)
        try:
            _validate(init_model(shape=112), img_size=(112, 112), batch_size=128) #128 Mem 6446MB
        except Exception as e:
            print(e)
        _validate(init_model(shape=384), img_size=(384, 384), batch_size=128)
        _validate(init_model(shape=512), img_size=(512, 512), batch_size=128)
        _validate(init_model(shape=640), img_size=(640, 640), batch_size=128)
    if True:
        _validate(init_model(shape=768), img_size=(768, 768), batch_size=84)
    if True:
        _validate(init_model(shape=1024), img_size=(1024, 1024), batch_size=(36 if model in ["deit"] else 48))
        _validate(init_model(shape=1120), img_size=(1120, 1120), batch_size=(24 if model in ["deit"] else 36))
        # _validate(init_model(shape=1280), img_size=(1280, 1280), batch_size=32)

def main_flops():
    from get_flops import fvcore_flop_count
    print("============ tiny ===================")
    init_vssm = partial(build_vssm, ckpt=os.path.join(os.path.dirname(__file__), "../../ckpts/vssmtiny/ckpt_epoch_292.pth"), only_backbone=True)
    init_swin = partial(build_mmpretrain_models, cfg="swin_tiny", ckpt=True, only_backbone=True, with_norm=True)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=True, only_backbone=True, with_norm=True)
    init_deit = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=True, with_norm=True)
    init_resnet = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=True, with_norm=True)
    for model in ["vssm", "swin", "convnext", "deit", "resnet"]:
        init_model = eval(f"init_{model}")
        for shape in [64, 112, 224, 384, 512, 640, 768, 1024, 1120, 1280]:
            try:
                params, flops = fvcore_flop_count(init_model(shape=shape), input_shape=(3, shape, shape))
                print(f"======== model {model} img_size {shape} params {params} flops {flops}", flush=True)
            except:
                pass
    print("============ small ===================")
    init_vssm = partial(build_vssm, ckpt=os.path.join(os.path.dirname(__file__), "../../ckpts/vssmsmall/ema_ckpt_epoch_238.pth"), depths=[2,2,27,2], only_backbone=True)
    init_swin = partial(build_mmpretrain_models, cfg="swin_small", ckpt=True, only_backbone=True, with_norm=True)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_small", ckpt=True, only_backbone=True, with_norm=True)
    init_deit = partial(build_mmpretrain_models, cfg="deit_base", ckpt=True, only_backbone=True, with_norm=True)
    init_resnet = partial(build_mmpretrain_models, cfg="resnet101", ckpt=True, only_backbone=True, with_norm=True)
    for model in ["vssm", "swin", "convnext", "deit", "resnet"]:
        init_model = eval(f"init_{model}")
        for shape in [64, 112, 224, 384, 512, 640, 768, 1024, 1120, 1280]:
            try:
                params, flops = fvcore_flop_count(init_model(shape=shape), input_shape=(3, shape, shape))
                print(f"======== model {model} img_size {shape} params {params} flops {flops}", flush=True)
            except:
                pass
    print("============ base ===================")
    init_vssm = partial(build_vssm, ckpt=os.path.join(os.path.dirname(__file__), "../../ckpts/vssmbase/ckpt_epoch_260.pth"), depths=[2,2,27,2], dims=128, only_backbone=True)
    init_swin = partial(build_mmpretrain_models, cfg="swin_base", ckpt=True, only_backbone=True, with_norm=True)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_base", ckpt=True, only_backbone=True, with_norm=True)
    init_replknet = partial(build_mmpretrain_models, cfg="replknet_base", ckpt=True, only_backbone=True, with_norm=True)
    for model in ["vssm", "swin", "convnext", "replknet"]:
        init_model = eval(f"init_{model}")
        for shape in [64, 112, 224, 384, 512, 640, 768, 1024, 1120, 1280]:
            try:
                params, flops = fvcore_flop_count(init_model(shape=shape), input_shape=(3, shape, shape))
                print(f"======== model {model} img_size {shape} params {params} flops {flops}", flush=True)
            except:
                pass


def main(model="replknet", action="tiny"):
    if action in ['tiny']:
        main_tiny(model)
    elif action in ['small']:
        main_small(model)
    elif action in ['base']:
        main_base(model)
    elif action in ['flops']:
        main_flops()

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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
    main(os.environ['SCALENET'], os.environ['ACTION'])

