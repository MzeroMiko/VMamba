#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# https://github.com/facebookresearch/moco/blob/main/main_lincls.py





import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def parse_options():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--epochs", default=30, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=30.0,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.0,
        type=float,
        metavar="W",
        help="weight decay (default: 0.)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--pretrained", default="", type=str, help="path to moco pretrained checkpoint"
    )
    args = parser.parse_args()

    print(args)
    return args


def get_feats_train_dataloader(features=[], targets=[], length=1281167, dim=768, batch_size=128, distributed=False, train=True):
    assert len(features) == len(targets)
    feats = torch.zeros((length, dim))
    tgts = torch.zeros((length))

    class fds(torch.utils.data.Dataset):
        def __init__(self):
            super().__init__()
        
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


def get_feats_eval_dataloader(features=[], targets=[], length=50000, dim=768, batch_size=128):
    assert len(features) == len(targets)
    feats = torch.zeros((length, dim))
    tgts = torch.zeros((length))

    class fds(torch.utils.data.Dataset):
        def __init__(self):
            super().__init__()
        
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




def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def main(args, outdir="."):
    model = nn.Linear(args.dim, args.num_classes, bias=True)

    model.weight.data.normal_(mean=0.0, std=0.01)
    model.bias.data.zero_()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )

    train_loader = get_feats_train_dataloader()
    val_loader = get_feats_eval_dataloader()

    for epoch in range(0, args.epochs):
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        top1 = AverageMeter("Acc@1", ":6.3f")
        top5 = AverageMeter("Acc@5", ":6.3f")
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch),
        )

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        validate(val_loader, model, criterion, args)

        torch.save({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join(outdir, f"ckpt_epoch_{epoch}.pth"))


if __name__ == "__main__":
    main()




    