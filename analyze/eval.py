import time
import tqdm
import torch
import torch.utils.data
import argparse
import os
import sys
import logging
from functools import partial
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import EncoderBlock
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import math
logging.basicConfig(level=logging.INFO)
logger = logging
from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


HOME = os.environ["HOME"].rstrip("/")
basicpath = os.path.abspath("../VMamba/analyze").rstrip("/")
basicpath = os.path.abspath(os.path.dirname(__file__)).rstrip("/")

# this mode will greatly inference the speed!
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from utils import ExtractFeatures, BuildModels
from analyze_for_vim import ExtraDev



def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


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


def get_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


def _validate(
    model: nn.Module = None, 
    freq=10, 
    amp=True, 
    img_size=224, 
    batch_size=128, 
    data_path="/dataset/ImageNet2012",
):
    class Args():
        AMP_ENABLE = amp
        PRINT_FREQ = freq
    config = Args()

    model.cuda().eval()
    model = torch.nn.parallel.DistributedDataParallel(model)
    _batch_size = batch_size
    while _batch_size > 0:
        try:
            _dataloader = get_dataloader(
                batch_size=_batch_size, 
                root=os.path.join(os.path.abspath(data_path), "val"),
                img_size=img_size,
                sequential=False,
            )
            logging.info(f"starting loop: img_size {img_size}; len(dataset) {len(_dataloader.dataset)}")
            validate(config, data_loader=_dataloader, model=model)
            break
        except:
            _batch_size = _batch_size // 2
            print(f"batch_size {_batch_size}", flush=True)


def _extract_feature(data_path="ImageNet_ILSVRC2012", start=0, end=200, step=-1, img_size=224, batch_size=16, train=True, aug=False):
    if False:
        resnet50 = BuildModels.build_resnet_mmpretrain(with_ckpt=True, remove_head=True, scale="r50", size=img_size).cuda().eval()
        deitsmall = BuildModels.build_deit_mmpretrain(with_ckpt=True, remove_head=True, scale="small", size=img_size).cuda().eval()
        vmambav0tiny = BuildModels.build_vmamba(with_ckpt=True, remove_head=True, scale="tv0").cuda().eval()
        vmambav2l5tiny = BuildModels.build_vmamba(with_ckpt=True, remove_head=True, scale="tv1").cuda().eval()
        vmambav2tiny = BuildModels.build_vmamba(with_ckpt=True, remove_head=True, scale="tv2").cuda().eval()
        convnexttiny = BuildModels.build_convnext(with_ckpt=True, remove_head=True, scale="tiny").cuda().eval()
        swintiny = BuildModels.build_swin_mmpretrain(with_ckpt=True, remove_head=True, scale="tiny", size=img_size).cuda().eval()
        hivittiny = BuildModels.build_hivit_mmpretrain(with_ckpt=True, remove_head=True, scale="tiny", size=img_size).cuda().eval()
        interntiny = BuildModels.build_intern(with_ckpt=True, remove_head=True, scale="tiny").cuda().eval()
        xcittiny = BuildModels.build_xcit(with_ckpt=True, remove_head=True, scale="tiny", size=img_size).cuda().eval()
        deitbase = BuildModels.build_deit_mmpretrain(with_ckpt=True, remove_head=True, scale="base", size=img_size).cuda().eval()

    if True:
        vims = ExtraDev.build_vim_for_throughput(with_ckpt=True, remove_head=True, size=img_size).cuda().eval()

    if True:
        if step > 0:
            starts = list(range(start, end, step))
            ends = [s + step for s in starts]
            assert ends[-1] >= end
            ends[-1] = end
            print(f"multiple ranges: {starts} {ends} ==============", flush=True)
        else:
            starts, ends = [start], [end]

        for s, e in zip(starts, ends):
            ExtractFeatures.extract_feature(
                backbones=dict(
                    # vmambav2tiny = vmambav2tiny,
                    # convnexttiny = convnexttiny,
                    # swintiny = swintiny,
                    # interntiny = interntiny,
                    # vmambav0tiny = vmambav0tiny,
                    # vmambav2l5tiny = vmambav2l5tiny,
                    # deitsmall = deitsmall,
                    # hivittiny = hivittiny,
                    # resnet50 = resnet50,
                    # xcittiny = xcittiny,
                    # deitbase = deitbase,
                    vims = vims,
                ), 
                dims=dict(
                    # vmambav2tiny = 768,
                    # convnexttiny = 768,
                    # swintiny = 768,
                    # interntiny = 768,
                    # vmambav0tiny = 768,
                    # vmambav2l5tiny = 768,
                    # deitsmall = 384,
                    # hivittiny = 384,
                    # resnet50 = 2048,
                    # xcittiny = 384,
                    # deitbase = 768,
                    vims = 384,
                ),
                batch_size=batch_size,
                img_size=img_size,
                data_path=data_path,
                ranges=(s, e),
                train=train,
                aug=aug,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="ImageNet_ILSVRC2012", help='path to dataset')
    parser.add_argument('--mode', type=str, default="", help='model name')
    parser.add_argument('--func', type=str, default="", help='function')
    parser.add_argument('--start', type=int, default=0, help='start range')
    parser.add_argument('--end', type=int, default=200, help='end range')
    parser.add_argument('--step', type=int, default=-1, help='step range')
    parser.add_argument('--size', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--val', action="store_true", help='...')
    parser.add_argument('--aug', action="store_true", help='...')
    args = parser.parse_args()
    print(args, flush=True)
    _extract_feature(args.data_path, args.start, args.end, args.step, args.size, args.batch_size, (not args.val), args.aug)


def run_code_dist_one(func):
    if torch.cuda.device_count() > 1:
        print("WARNING!!!  acc score would be inaccurate if num_procs > 1, as sampler always pads the dataset")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # print(torch.cuda.device_count())
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
    run_code_dist_one(main)

