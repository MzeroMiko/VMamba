import argparse
import os
import time
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
import torch.utils.data
from timm.utils import accuracy, AverageMeter
import logging
logger = logging
HOME = os.environ["HOME"].rstrip("/")


def parse_options():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("-b", "--batch-size", default=4096, type=int, dest="batch_size")
    parser.add_argument("--lr", "--learning-rate", default=30.0, type=float, dest="lr")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=0.0, type=float, dest="weight_decay")
    parser.add_argument("--reinit", action="store_true")
    parser.add_argument("-e", "--evaluate", action="store_true", dest="evaluate")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--size", default=224, type=int, help="img size")
    parser.add_argument("--name", default="all", type=str, help="model name")
    args = parser.parse_args()
    print(args)
    return args


def get_feats_train_dataloader(features, length=1281167,batch_size=128, distributed=False):
    feats = torch.load(open(features, "rb"))
    feats, tgts = feats["features"], feats["targets"].long()
    assert feats.shape[0] == length
    assert tgts.shape[0] == length

    class fds(torch.utils.data.Dataset):
        def __len__(self):
            return length
        
        def __getitem__(self, index):
            return feats[index], tgts[index]

    dataset_train = fds()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    ) if distributed else None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        shuffle=(not distributed),
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    return data_loader_train


def get_feats_eval_dataloader(features, length=50000, batch_size=128):
    feats = torch.load(open(features, "rb"))
    feats, tgts = feats["features"], feats["targets"].long()
    assert feats.shape[0] == length
    assert tgts.shape[0] == length
    
    class fds(torch.utils.data.Dataset):
        def __len__(self):
            return length
        
        def __getitem__(self, index):
            return feats[index], tgts[index]

    dataset_val = fds()
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=None,
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return data_loader_val


# WARNING!!!  acc score would be inaccurate if num_procs > 1, as sampler always pads the dataset
# copied from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
@torch.no_grad()
def validate(data_loader, model, AMP_ENABLE=True, verbose=True):
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
        with torch.cuda.amp.autocast(enabled=AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if verbose: 
        print(f'* Loss {loss_meter.avg:.4f} Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}', flush=True)
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def train(model, args, features_train, features_val, seed=0, state_dict=None, reinit=False, outdir="/tmp", val=False, lr=0.05, verbose=True):
    batch_size = args.batch_size
    print(args, dict(model=model, lr=lr, verbose=verbose, seed=seed, reinit=reinit), flush=True)
    
    assert isinstance(model, torch.nn.Linear)
    # model = torch.nn.Linear(args.dim, args.num_classes, bias=True)

    train_loader = get_feats_train_dataloader(features_train, batch_size=batch_size, length=1281167)
    val_loader = get_feats_eval_dataloader(features_val, batch_size=batch_size, length=50000)

    model = torch.nn.Sequential(OrderedDict(fc = model,)).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    # optimizer = torch.optim.AdamW(model.parameters(), lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=0)

    if state_dict is not None:
        model.fc.load_state_dict(state_dict)
        validate(val_loader, model)
    
    if seed is not None:
        assert isinstance(seed, int)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True

    if reinit:
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        validate(val_loader, model, verbose=True)
    
    if val:
        return

    maxacc1 = [0, 0, 0, 0]
    for epoch in range(0, args.epochs):
        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        model.train()
        for idx, (images, target) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            acc1_meter.update(acc1.item(), images.size(0))
            acc5_meter.update(acc5.item(), images.size(0))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if verbose:
            print(
                f'Train[{epoch}/{args.epochs} : {len(train_loader)}]: '
                f'Loss {loss_meter.avg:.4f} '
                f'Acc@1 {acc1_meter.avg:.3f} '
                f'Acc@5 {acc5_meter.avg:.3f} ', flush=True)

        acc1, acc5, loss = validate(val_loader, model, verbose=verbose)
        if acc1 > maxacc1[0]:
            maxacc1 = [acc1, acc5, loss, epoch]
    print(f"max acc: {maxacc1[0:2]}, loss: {maxacc1[2]}, epoch {maxacc1[3]}", flush=True)

    torch.save({
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
    }, os.path.join(outdir, f"ckpt_epoch_{args.epochs}.pth"))


if __name__ == "__main__":
    args = parse_options()

    vmambav2tiny = dict(
        name = "vmambav2tiny",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["classifier.head.weight"],
            "bias": sd["model"]["classifier.head.bias"],
        } 
    )

    vmambav2l5tiny = dict(
        name = "vmambav2l5tiny",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230/vssm1_tiny_0230_ckpt_epoch_262.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["classifier.head.weight"],
            "bias": sd["model"]["classifier.head.bias"],
        } 
    )

    vmambav0tiny = dict(
        name = "vmambav0tiny",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["head.weight"],
            "bias": sd["model"]["head.bias"],
        } 
    )

    resnet50 = dict(
        name = "resnet50",
        model = nn.Linear(2048, 1000, bias=True),
        ckpt = f"{HOME}/.cache/torch/hub/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth",
        state_dict = lambda sd: {
            "weight": sd["state_dict"]["head.fc.weight"],
            "bias": sd["state_dict"]["head.fc.bias"],
        } 
    )

    deitsmall = dict(
        name = "deitsmall",
        model = nn.Linear(384, 1000, bias=True),
        ckpt = f"{HOME}/.cache/torch/hub/checkpoints/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth",
        state_dict = lambda sd: {
            "weight": sd["state_dict"]["head.layers.head.weight"],
            "bias": sd["state_dict"]["head.layers.head.bias"],
        } 
    )

    convnexttiny = dict(
        name = "convnexttiny",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/packs/ckpts/convnext_tiny_1k_224_ema.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["head.weight"],
            "bias": sd["model"]["head.bias"],
        } 
    )

    swintiny = dict(
        name = "swintiny",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/.cache/torch/hub/checkpoints/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth",
        state_dict = lambda sd: {
            "weight": sd["state_dict"]["head.fc.weight"],
            "bias": sd["state_dict"]["head.fc.bias"],
        } 
    )

    hivittiny = dict(
        name = "hivittiny",
        model = nn.Linear(384, 1000, bias=True),
        ckpt = f"{HOME}/packs/ckpts/hivit-tiny-p16_8xb128_in1k/epoch_295.pth",
        state_dict = lambda sd: {
            "weight": sd["state_dict"]["head.fc.weight"],
            "bias": sd["state_dict"]["head.fc.bias"],
        } 
    )

    interntiny = dict(
        name = "interntiny",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/packs/ckpts/internimage_t_1k_224.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["head.weight"],
            "bias": sd["model"]["head.bias"],
        } 
    )

    xcittiny = dict(
        name = "xcittiny",
        model = nn.Linear(384, 1000, bias=True),
        ckpt = f"{HOME}/packs/ckpts/xcit_small_12_p16_224.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["head.weight"],
            "bias": sd["model"]["head.bias"],
        }  
    )

    deitbase = dict(
        name = "deitbase ",
        model = nn.Linear(768, 1000, bias=True),
        ckpt = f"{HOME}/.cache/torch/hub/checkpoints/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth",
        state_dict = lambda sd: {
            "weight": sd["state_dict"]["head.layers.head.weight"],
            "bias": sd["state_dict"]["head.layers.head.bias"],
        } 
    )

    vims = dict(
        name = "vims",
        model = nn.Linear(384, 1000, bias=True),
        ckpt = f"{HOME}/packs/ckpts/vim_s_midclstok_80p5acc.pth",
        state_dict = lambda sd: {
            "weight": sd["model"]["head.weight"],
            "bias": sd["model"]["head.bias"],
        }         
    )

    names = {}
    for col in [vmambav2tiny, vmambav2l5tiny, vmambav0tiny, swintiny, convnexttiny, hivittiny, deitsmall, resnet50, interntiny, xcittiny, deitbase, vims]:
        names.update({col["name"]: col})
        size = 224
        model = col["model"]
        feature_train = f"{HOME}/ckpts/feats/merge{size}/{col['name']}_sz{size}_train.pth"
        feature_val = f"{HOME}/ckpts/feats/merge{size}/{col['name']}_sz{size}_val.pth"
        state_dict = col["state_dict"](torch.load(col["ckpt"], map_location=torch.device("cpu")))

    if args.name == "all":
        # for col in [vmambav2tiny, vmambav2l5tiny, vmambav0tiny, swintiny, convnexttiny, hivittiny, deitsmall, resnet50, interntiny, xcittiny, vims]:
        for col in [vims]:
            for size, lr in zip([224, 288, 384, 512, 640, 768, 1024], [0.05, 0.05, 0.05, 0.2, 0.5, 0.5, 0.5]):
                model = col["model"]
                feature_train = f"{HOME}/ckpts/feats/merge{size}/{col['name']}_sz{size}_train.pth"
                feature_val = f"{HOME}/ckpts/feats/merge{size}/{col['name']}_sz{size}_val.pth"
                state_dict = col["state_dict"](torch.load(col["ckpt"], map_location=torch.device("cpu")))
                train(
                    model=model, args=args, features_train=feature_train, features_val=feature_val,
                    state_dict=state_dict, 
                    reinit=args.reinit,
                    val=args.evaluate,
                    lr=lr,
                    verbose=False,
                )
    else:
        size = args.size
        col = names[args.name]
        model = col["model"]
        feature_train = f"{HOME}/ckpts/feats/merge{size}/{col['name']}_sz{size}_train.pth"
        feature_val = f"{HOME}/ckpts/feats/merge{size}/{col['name']}_sz{size}_val.pth"
        state_dict = col["state_dict"](torch.load(col["ckpt"], map_location=torch.device("cpu")))
        train(
            model=model, args=args, features_train=feature_train, features_val=feature_val,
            state_dict=state_dict, 
            reinit=args.reinit,
            val=args.evaluate,
            lr = args.lr,
        )




    