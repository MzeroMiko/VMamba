import os
from functools import partial
from typing import Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from PIL import Image
import math

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

vmamba = import_abspy(
    "vmamba", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/models"),
)
VSSM: nn.Module = vmamba.VSSM
SS2D: nn.Module = vmamba.SS2D
VSSBlock: nn.Module = vmamba.VSSBlock
Mlp: nn.Module = vmamba.Mlp
gMlp: nn.Module = vmamba.gMlp
DropPath: nn.Module = vmamba.DropPath
SelectiveScanOflex: nn.Module = vmamba.SelectiveScanOflex
CrossScanTriton: nn.Module = vmamba.CrossScanTriton
CrossMergeTriton: nn.Module = vmamba.CrossMergeTriton
CrossScanTriton1b1: nn.Module = vmamba.CrossScanTriton1b1

this_path = os.path.dirname(os.path.abspath(__file__))

from erf import visualize
visualize_attnmap = visualize.visualize_attnmap
visualize_attnmaps = visualize.visualize_attnmaps


def get_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
    from torch.utils.data import SequentialSampler, DistributedSampler, DataLoader
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(root, transform=transform)
    if sequential:
        sampler = SequentialSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)
    
    data_loader = DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


def denormalize(image: torch.Tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    if len(image.shape) == 2:
        image = (image.cpu() * 255).to(torch.uint8).numpy()
    elif len(image.shape) == 3:
        C, H, W = image.shape
        image = image.cpu() * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    image = Image.fromarray(image)
    return image


@torch.no_grad()
def attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, with_ws=True, with_dt=False, only_ws=False, ret="all", ratio=1, verbose=False):
    printlog = print if verbose else lambda *args, **kwargs: None
    printlog(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)

    B, G, N, L = Bs.shape
    GD, N = As.shape
    D = GD // G
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    mask = torch.tril(dts.new_ones((L, L)))
    dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
    dw_logs = As.view(G, D, N)[None, :, :, None] * dts[:,:,:,None,:] # (B, G, D, N, L)
    ws = torch.cumsum(dw_logs, dim=-1).exp()

    if only_ws:
        Qs, Ks = ws, 1 / ws.clamp(min=1e-20)
    else:
        Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:]
        if with_ws:
            Qs, Ks = Qs * ws, Ks / ws.clamp(min=1e-20)
    if with_dt:
        Ks = Ks * dts.view(B, G, D, 1, L)

    printlog(ws.shape, Qs.shape, Ks.shape)
    printlog("Bs", Bs.max(), Bs.min(), Bs.abs().min())
    printlog("Cs", Cs.max(), Cs.min(), Cs.abs().min())
    printlog("ws", ws.max(), ws.min(), ws.abs().min())
    printlog("Qs", Qs.max(), Qs.min(), Qs.abs().min())
    printlog("Ks", Ks.max(), Ks.min(), Ks.abs().min())
    _Qs, _Ks = Qs.view(-1, N, L), Ks.view(-1, N, L)
    attns = (_Qs.transpose(1, 2) @ _Ks).view(B, G, -1, L, L)
    attns = attns.mean(dim=2) * mask

    attn0 = attns[:, 0, :].view(B, -1, L, L)
    attn1 = attns[:, 1, :].view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)
    attn2 = attns[:, 2, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
    attn3 = attns[:, 3, :].view(-1, L, L).flip(dims=[-2]).flip(dims=[-1]).contiguous().view(B, -1, L, L)
    attn3 = attn3.view(-1, H, W, H, W).permute(0, 2, 1, 4, 3).contiguous().view(B, -1, L, L)

    if ret in ["ao0"]:
        attn = attns[:, 0, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["ao1"]:
        attn = attns[:, 1, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["ao2"]:
        attn = attns[:, 2, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["ao3"]:
        attn = attns[:, 3, :].view(B, -1, L, L).mean(dim=1)
    elif ret in ["a0"]:
        attn = attn0.mean(dim=1)
    elif ret in ["a1"]:
        attn = attn1.mean(dim=1)
    elif ret in ["a2"]:
        attn = attn2.mean(dim=1)
    elif ret in ["a3"]:
        attn = attn3.mean(dim=1)
    elif ret in ["a0a2"]:
        attn = (attn0 + attn2).mean(dim=1)
    elif ret in ["a1a3"]:
        attn = (attn1 + attn3).mean(dim=1)
    elif ret in ["a0a1"]:
        attn = (attn0 + attn1).mean(dim=1)
    elif ret in ["all"]:
        attn = (attn0 + attn1 + attn2 + attn3).mean(dim=1)
    return ratio * attn[bidx, :, :]


def add_hook(model: nn.Module):
    ss2ds = []
    for layer in model.layers:
        _ss2ds = []
        for blk in layer.blocks:
            ss2d = blk.op
            setattr(ss2d, "__DEBUG__", True)
            _ss2ds.append(ss2d)
        ss2ds.append(_ss2ds)
    return model, ss2ds


def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith("backbone."):
            new_state_dict[k[len("backbone."):]] = state_dict[k]
    return new_state_dict


def visual_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs", H=56, W=56, front_point=(0.5, 0.5), front_back=(0.7, 0.8), showpath=os.path.join(this_path, "show")):
    kwargs = dict(with_ws=with_ws, with_dt=with_dt, only_ws=only_ws, ratio=ratio)
    visualize_attnmap(
        attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs),
        savefig=f"{showpath}/{tag}_merge.jpg"
    )
    visualize_attnmap(torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)).view(H, W), savefig=f"{showpath}/{tag}_attn_diag.jpg") # self attention
    # visualize_attnmap(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(front_point[0] * H * W + front_point[1] * W)].view(H, W), savefig=f"{showpath}/{tag}_attn_front.jpg") # front attention
    # visualize_attnmap(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(front_back[0] * H * W + front_back[1] * W)].view(H, W), savefig=f"{showpath}/{tag}_attn_back.jpg") # back attention
    visualize_attnmaps([
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao0", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao1", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao2", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao3", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs), ""),
    ], rows=1, savefig=f"{showpath}/{tag}_scan0.jpg", fontsize=60)    
    visualize_attnmaps([
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao0", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao1", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao2", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="ao3", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs), ""),
    ], rows=2, savefig=f"{showpath}/{tag}_scan.jpg", fontsize=60)
    visualize_attnmaps([
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[int(H * W / 3 * 2)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a0", **kwargs)[-1].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[int(H * W / 3 * 2)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a1", **kwargs)[-1].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[int(H * W / 3 * 2)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a2", **kwargs)[-1].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[0].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[int(H * W / 3)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[int(H * W / 3 * 2)].view(H, W), ""),
        (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="a3", **kwargs)[-1].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[0].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(H * W / 3)].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[int(H * W / 3 * 2)].view(H, W), ""),
        # (attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, ret="all", **kwargs)[-1].view(H, W), ""),
    ], rows=4, dpi=200, savefig=f"{showpath}/{tag}_scan_procedure.jpg", fontsize=100)


def main():
    vssm: nn.Module = VSSM(
        depths=[2, 2, 5, 2], 
        dims=[96, 192, 384, 768], 
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        # norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
    ).cuda().eval()
    vssm.load_state_dict(convert_state_dict(torch.load(open("/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/detection/mask_rcnn_vssm_fpn_coco_tiny/mask_rcnn_vssm_fpn_coco_tiny_epoch_12.pth", "rb"), map_location="cpu")["state_dict"]), strict=False)
    vssm, ss2ds = add_hook(vssm)
    showpath = os.path.join(this_path, "show")

    data = get_dataloader(batch_size=128, root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', sequential=True, img_size=448)
    dataset = data.dataset
    img, label = dataset[512]
    img, label = dataset[509]
    with torch.no_grad():
        out = vssm(img[None].cuda())
    print(out.argmax().item(), label)
    denormalize(img).save(f"{showpath}/imori.jpg")

    regs = getattr(ss2ds[-2][-1], "__data__")
    As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
    us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
    ys, oy = regs["ys"], regs["y"]
    print(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
    B, G, N, L = Bs.shape
    GD, N = As.shape
    D = GD // G
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    visual_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs", H=H, W=W)
    visual_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=True, with_dt=False, only_ws=True, ratio=1, tag="ws", H=H, W=W)


if __name__ == "__main__":
    main()

