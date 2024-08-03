import os
import logging
import sys
import time
import math
from functools import partial
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from collections import OrderedDict
import cv2
import PIL
import tqdm
from PIL import Image
import os
import sys

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from functools import partial
from typing import Callable, Tuple, Union, Tuple, Union, Any
from collections import defaultdict

HOME = os.environ["HOME"].rstrip("/")

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


def get_dataset(root="./val", img_size=224, ret="", crop=True, single_image=False):
    from torch.utils.data import SequentialSampler, DistributedSampler, DataLoader
    size = int((256 / 224) * img_size) if crop else int(img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ])
    if single_image:
        class ds(datasets.ImageFolder):
            def __init__(self, img, transform):
                self.transform = transform
                self.target_transform = None
                self.loader = datasets.folder.default_loader
                self.samples = [(img, 0)]
                self.targets = [0]
                self.classes = ["none"]
                self.class_to_idx = {"none": 0}
        dataset = ds(root, transform=transform)
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
        if ret in dataset.classes:
            print(f"found target {ret}", flush=True)
            target = dataset.class_to_idx[ret]
            dataset.samples =  [s for s in dataset.samples if s[1] == target]
            dataset.targets = [s for s in dataset.targets if s == target]
            dataset.classes = [ret]
            dataset.class_to_idx = {ret: target}
    return dataset


def show_mask_on_image(img: torch.Tensor, mask: torch.Tensor, mask_norm=True):
    H, W, C = img.shape
    mH, mW = mask.shape
    mask = torch.nn.functional.interpolate(mask[None, None], (H, W), mode="bilinear")[0, 0]
    if mask_norm:
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    img = img.clamp(min=0, max=1).cpu().numpy()
    mask = mask.clamp(min=0, max=1).cpu().numpy()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    return heatmap
    return np.uint8(255 * cam)


def get_val_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
    import torch.utils.data
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


class visualize:
    @staticmethod
    def get_colormap(name):
        import matplotlib as mpl
        """Handle changes to matplotlib colormap interface in 3.6."""
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return mpl.cm.get_cmap(name)

    @staticmethod
    def draw_image_grid(image: Image, grid=[(0, 0, 1, 1)], **kwargs):
        # grid[0]: (x,y,w,h)
        default = dict(fill=None, outline='red', width=3)
        default.update(kwargs)
        assert isinstance(grid, list) and isinstance(grid[0], tuple) and len(grid[0]) == 4
        from PIL import ImageDraw
        a = ImageDraw.ImageDraw(image)
        for g in grid:
            a.rectangle([(g[0], g[1]), (g[0] + g[2], g[1] + g[3])], **default)
        return image

    @staticmethod
    def visualize_attnmap(attnmap, savefig="", figsize=(18, 16), cmap=None, sticks=True, dpi=400, fontsize=35, colorbar=True, **kwargs):
        import matplotlib.pyplot as plt
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        # if isinstance(imgori, torch.Tensor):
        #     imgori = imgori.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        im = ax.imshow(attnmap, cmap=cmap)
        # ax.set_title(title)
        if not sticks:
            ax.set_axis_off()
        if colorbar:
            cbar = ax.figure.colorbar(im, ax=ax)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()

    @staticmethod
    def visualize_attnmaps(attnmaps, savefig="", figsize=(18, 16), rows=1, cmap=None, dpi=400, fontsize=35, linewidth=2, **kwargs):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        vmin = min([np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        vmax = max([np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize=(cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(rows, cols, squeeze=False, sharex="all", sharey="all", figsize=figsize, dpi=dpi)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                im = axs[i, j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
                axs[i, j].set_title(title)
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                print(title, "max", np.max(image), "min", np.min(image), end=" | ")
            print("")
        axs[0, 0].figure.colorbar(im, ax=axs)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()
        print("")

    @staticmethod
    def seanborn_heatmap(
        data, *,
        vmin=None, vmax=None, cmap=None, center=None, robust=False,
        annot=None, fmt=".2g", annot_kws=None,
        linewidths=0, linecolor="white",
        cbar=True, cbar_kws=None, cbar_ax=None,
        square=False, xticklabels="auto", yticklabels="auto",
        mask=None, ax=None,
        **kwargs
    ):
        from matplotlib import pyplot as plt
        from seaborn.matrix import _HeatMapper
        # Initialize the plotter object
        plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
                            annot_kws, cbar, cbar_kws, xticklabels,
                            yticklabels, mask)

        # Add the pcolormesh kwargs here
        kwargs["linewidths"] = linewidths
        kwargs["edgecolor"] = linecolor

        # Draw the plot and return the Axes
        if ax is None:
            ax = plt.gca()
        if square:
            ax.set_aspect("equal")
        plotter.plot(ax, cbar_ax, kwargs)
        mesh = ax.pcolormesh(plotter.plot_data, cmap=plotter.cmap, **kwargs)
        return ax, mesh

    @classmethod
    def visualize_snsmap(cls, attnmap, savefig="", figsize=(18, 16), cmap=None, sticks=True, dpi=80, fontsize=35, linewidth=2, **kwargs):
        import matplotlib.pyplot as plt
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        _, mesh = cls.seanborn_heatmap(attnmap, xticklabels=sticks, yticklabels=sticks, cmap=cmap, linewidths=0,
                center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
        cb = ax.figure.colorbar(mesh, ax=ax)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()

    @classmethod
    def visualize_snsmaps(cls, attnmaps, savefig="", figsize=(18, 16), rows=1, cmap=None, sticks=True, dpi=80, fontsize=35, linewidth=2, **kwargs):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        vmin = min([np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        vmax = max([np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a)) for a, t in attnmaps])
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize=(cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(rows, cols, squeeze=False, sharex="all", sharey="all", figsize=figsize, dpi=dpi)
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                _, im = cls.seanborn_heatmap(image, xticklabels=sticks, yticklabels=sticks, 
                                             vmin=vmin, vmax=vmax, cmap=cmap,
                                             center=0, annot=False, ax=axs[i, j], 
                                             cbar=False, annot_kws={"size": 24}, fmt='.2f')
                axs[i, j].set_title(title)
        cb = axs[0, 0].figure.colorbar(im, ax=axs)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()


#  used for visualizing effective receiptive field 
class EffectiveReceiptiveField:
    @staticmethod
    def simpnorm(data):
        data = np.power(data, 0.2)
        data = data / np.max(data)
        return data

    @staticmethod
    def get_rectangle(data, thresh):
        h, w = data.shape
        all_sum = np.sum(data)
        for i in range(1, h // 2):
            selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
            area_sum = np.sum(selected_area)
            if area_sum / all_sum > thresh:
                return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
        return None, None

    @staticmethod
    def get_input_grad(model, samples, square=True):
        outputs = model(samples)
        out_size = outputs.size()
        if square:
            assert out_size[2] == out_size[3], out_size
        central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    @classmethod
    def get_input_grad_avg(cls, model: nn.Module, size=1024, data_path="ImageNet", num_images=50, norms=lambda x:x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
        import tqdm
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader, RandomSampler
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        dataset = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)
        data_loader_val = DataLoader(dataset, sampler=RandomSampler(dataset), pin_memory=True)

        meter = AverageMeter()
        model.cuda().eval()
        for _, (samples, _) in tqdm.tqdm(enumerate(data_loader_val)):
            if meter.count == num_images:
                break
            samples = samples.cuda(non_blocking=True).requires_grad_()
            contribution_scores = cls.get_input_grad(model, samples)
            if np.isnan(np.sum(contribution_scores)):
                print("got nan | ", end="")
                continue
            else:
                meter.update(contribution_scores)
        return norms(meter.avg)


# used for visualizing the attention of mamba
class AttnMamba:
    @staticmethod
    def convert_state_dict_from_mmdet(state_dict):
        new_state_dict = OrderedDict()
        for k in state_dict:
            if k.startswith("backbone."):
                new_state_dict[k[len("backbone."):]] = state_dict[k]
        return new_state_dict

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value

    @staticmethod
    @torch.no_grad()
    def attnmap_mamba(regs, mode="CB", ret="all", absnorm=0, scale=1, verbose=False, device=None):
        printlog = print if verbose else lambda *args, **kwargs: None
        print(f"attn for mode={mode}, ret={ret}, absnorm={absnorm}, scale={scale}", flush=True)

        _norm = lambda x: x
        if absnorm == 1:
            _norm = lambda x: ((x - x.min()) / (x.max() - x.min()))
        elif absnorm == 2:
            _norm = lambda x: (x.abs() / x.abs().max())

        As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
        us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
        ys, oy = regs["ys"], regs["y"]
        H, W = regs["H"], regs["W"]
        printlog(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
        B, G, N, L = Bs.shape
        GD, N = As.shape
        D = GD // G
        H, W = int(math.sqrt(L)), int(math.sqrt(L))
        if device is not None:
            As, Bs, Cs, Ds, us, dts, delta_bias, ys, oy = As.to(device), Bs.to(device), Cs.to(device), Ds.to(device), us.to(device), dts.to(device), delta_bias.to(device), ys.to(device), oy.to(device)

        mask = torch.tril(dts.new_ones((L, L)))
        dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
        dw_logs = As.view(G, D, N)[None, :, :, :, None] * dts[:,:,:,None,:] # (B, G, D, N, L)
        ws = torch.cumsum(dw_logs, dim=-1).exp()

        if mode == "CB":
            Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:]
        elif mode == "CBdt":
            Qs, Ks = Cs[:,:,None,:,:], Bs[:,:,None,:,:] * dts.view(B, G, D, 1, L)
        elif mode == "CwBw":
            Qs, Ks = Cs[:,:,None,:,:] * ws, Bs[:,:,None,:,:] / ws.clamp(min=1e-20)
        elif mode == "CwBdtw":
            Qs, Ks = Cs[:,:,None,:,:] * ws, Bs[:,:,None,:,:]  * dts.view(B, G, D, 1, L) / ws.clamp(min=1e-20)
        elif mode == "ww":
            Qs, Ks = ws, 1 / ws.clamp(min=1e-20)
        else:
            raise NotImplementedError

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

        # ao0, ao1, ao2, ao3: attntion in four directions without rearrange
        # a0, a1, a2, a3: attntion in four directions with rearrange
        # a0a2, a1a3, a0a1: combination of "a0, a1, a2, a3"
        # all: combination of all "a0, a1, a2, a3"
        if ret in ["ao0"]:
            attn = _norm(attns[:, 0, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao1"]:
            attn = _norm(attns[:, 1, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao2"]:
            attn = _norm(attns[:, 2, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["ao3"]:
            attn = _norm(attns[:, 3, :]).view(B, -1, L, L).mean(dim=1)
        elif ret in ["a0"]:
            attn = _norm(attn0).mean(dim=1)
        elif ret in ["a1"]:
            attn = _norm(attn1).mean(dim=1)
        elif ret in ["a2"]:
            attn = _norm(attn2).mean(dim=1)
        elif ret in ["a3"]:
            attn = _norm(attn3).mean(dim=1)
        elif ret in ["all"]:
            attn = _norm((attn0 + attn1 + attn2 + attn3)).mean(dim=1)
        elif ret in ["nall"]:
            attn = (_norm(attn0) + _norm(attn1) + _norm(attn2) + _norm(attn3)).mean(dim=1) / 4.0
        else:
            raise NotImplementedError(f"{ret} is not allowed")
        attn = (scale * attn).clamp(max=attn.max())
        return attn[0], H, W

    @classmethod
    @torch.no_grad()
    def get_attnmap_mamba(cls, ss2ds, stage=-1, mode="", verbose=False, raw_attn=False, block_id=0, scale=1, device=None):
        mode1 = mode.split("_")[-1]
        mode = mode[:-(len(mode1) + 1)]
        
        absnorm = 0
        tag, mode = cls.checkpostfix("_absnorm", mode)
        absnorm = 2 if tag else absnorm
        tag, mode = cls.checkpostfix("_norm", mode)
        absnorm = 1 if tag else absnorm

        if raw_attn:
            ss2d = ss2ds if not isinstance(ss2ds, list) else ss2ds[stage][block_id]
            regs = getattr(ss2d, "__data__")
            attn, H, W = cls.attnmap_mamba(regs, mode=mode1, ret=mode, absnorm=absnorm, verbose=verbose, scale=scale)
            return attn

        allrolattn = None
        for k in range(len(ss2ds[stage])):
            regs = getattr(ss2ds[stage][k], "__data__")
            attn, H, W = cls.attnmap_mamba(regs, mode=mode1, ret=mode, absnorm=absnorm, verbose=verbose, scale=scale)
            L = H * W
            assert attn.shape == (L, L)
            assert attn.max() <= 1
            assert attn.min() >= 0
            rolattn = 0.5 * (attn.cpu() + torch.eye(L))
            rolattn = rolattn / rolattn.sum(-1)
            allrolattn = (rolattn @ allrolattn) if allrolattn is not None else rolattn
        return allrolattn
        

# used for test throughput
class Throughput:
    # default no amp in testing tp
    # copied from swin_transformer
    @staticmethod
    @torch.no_grad()
    def throughput(data_loader, model, logger=logging):
        model.eval()
        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @staticmethod
    @torch.no_grad()
    def throughputamp(data_loader, model, logger=logging):
        model.eval()

        for idx, (images, _) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast():
                    model(images)
            torch.cuda.synchronize()
            logger.info(f"throughput averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast():
                    model(images)
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @staticmethod
    def testfwdbwd(data_loader, model, logger, amp=True):
        model.cuda().train()
        criterion = torch.nn.CrossEntropyLoss()

        for idx, (images, targets) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            batch_size = images.shape[0]
            for i in range(50):
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(images)
                    loss = criterion(out, targets)
                    loss.backward()
            torch.cuda.synchronize()
            logger.info(f"testfwdbwd averaged with 30 times")
            torch.cuda.reset_peak_memory_stats()
            tic1 = time.time()
            for i in range(30):
                with torch.cuda.amp.autocast(enabled=amp):
                    out = model(images)
                    loss = criterion(out, targets)
                    loss.backward()
            torch.cuda.synchronize()
            tic2 = time.time()
            logger.info(f"batch_size {batch_size} testfwdbwd {30 * batch_size / (tic2 - tic1)}")
            logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
            return

    @classmethod    
    def testall(cls, model, dataloader=None, data_path="", img_size=224, _batch_size=128, with_flops=True, inference_only=False):
        from fvcore.nn import parameter_count
        torch.cuda.empty_cache()
        model.cuda().eval()
        if with_flops:
            try:
                FLOPs.fvcore_flop_count(model, input_shape=(3, img_size, img_size), show_arch=False)
            except Exception as e:
                print("ERROR:", e, flush=True)
        print(parameter_count(model)[""], sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
        if dataloader is None:
            dataloader = get_val_dataloader(
                batch_size=_batch_size, 
                root=os.path.join(os.path.abspath(data_path), "val"),
                img_size=img_size,
            )
        cls.throughput(data_loader=dataloader, model=model, logger=logging)
        if inference_only:
            return
        PASS = False
        batch_size = _batch_size
        while (not PASS) and (batch_size > 0):
            try:
                _dataloader = get_val_dataloader(
                    batch_size=batch_size, 
                    root=os.path.join(os.path.abspath(data_path), "val"),
                    img_size=img_size,
                )
                cls.testfwdbwd(data_loader=_dataloader, model=model, logger=logging)
                cls.testfwdbwd(data_loader=_dataloader, model=model, logger=logging, amp=False)
                PASS = True
            except:
                batch_size = batch_size // 2
                print(f"batch_size {batch_size}", flush=True)



# used for extract features
class ExtractFeatures:
    @staticmethod
    def get_list_dataset(*args, **kwargs):
        class DatasetList:
            def __init__(self, batch_size=16, root="train/", img_size=224, weak_aug=False):
                self.batch_size = int(batch_size)
                transform, transform_waug = self.get_transform(img_size)
                self.transform = transform_waug if weak_aug else transform
                self.dataset = datasets.ImageFolder(root, transform=self.transform)
                
                self.num_data = int(len(self.dataset))
                self.num_batches = math.ceil(self.num_data / self.batch_size)
                print(f"weak aug: {weak_aug} =========================", flush=True)

            @staticmethod
            def get_transform(img_size=224):
                size = int((256 / 224) * img_size)
                transform = transforms.Compose([
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
                transform_waug = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                ])
                return transform, transform_waug

            def __len__(self):
                return self.num_batches

            def __getitem__(self, idx):
                start = idx * self.batch_size
                end = min(start + self.batch_size, self.num_data)
                data = [self.dataset[i] for i in range(start, end)]
                images = torch.stack([img for img, tgt in data])
                targets = torch.stack([torch.tensor(tgt) for img, tgt in data])
                if len(images) < self.batch_size:
                    _images = torch.zeros((self.batch_size, *data[0][0].shape))
                    _targets = -1 * torch.ones((self.batch_size,))
                    _images[:len(images)] = images
                    _targets[:len(images)] = targets
                    return _images, _targets
                return images, targets
        return DatasetList(*args, **kwargs)

    @classmethod
    def extract_feature(
        cls,
        backbones=dict(), # dict(name=model)
        batch_size=16, 
        img_size=1024,  
        data_path="ImageNet_ILSVRC2012", 
        amp_disable=False, 
        dims=dict(),  # dict(name=dim)
        outdir=os.path.join(HOME, "ckpts/feats/unmerge/"),
        ranges=[0, 1000],
        train=True,
        aug=False,
    ):
        root = os.path.join(data_path, "./train") if train else os.path.join(data_path, "./val")
        datasetlist = cls.get_list_dataset(batch_size, root=root, img_size=img_size, weak_aug=aug)

        ranges = list(ranges)
        if ranges[1] <= 0:
            ranges[1] = len(datasetlist)
        ranges[1] = min(ranges[1], len(datasetlist))
        assert len(ranges) == 2 and ranges[1] > ranges[0], f"{ranges}"
        outbatches = ranges[1] - ranges[0]
        outdir = os.path.join(outdir, f"sz{img_size}_bs{batch_size}_range{ranges[0]}_{ranges[1]}" + ("" if train else "_val"))
        os.makedirs(outdir, exist_ok=True)
        backbones = {
            name: torch.nn.parallel.DistributedDataParallel(model.cuda().eval())
            for name, model in backbones.items()
        }
        feats = {
            name: torch.zeros((outbatches, batch_size, dim))
            for name, dim in dims.items()
        }
        all_targets = torch.zeros((outbatches, batch_size))

        print("=" * 50, flush=True)
        print(f"using backbones {backbones.keys()}", flush=True)
        print(f"batch_size {batch_size} img_size {img_size} ranges {ranges} max_range {0} {len(datasetlist)}", flush=True)

        with torch.no_grad():
            for i, idx in enumerate(tqdm.tqdm(range(ranges[0], ranges[1]))):
                images, targets = datasetlist[idx]
                images = images.cuda(non_blocking=True)
                all_targets[i] = targets.detach().cpu()
                for name, model in backbones.items():
                    with torch.cuda.amp.autocast(enabled=(not amp_disable)):
                        feats[name][i] = model(images).detach().cpu()
            
            for name, model in backbones.items():
                na = f"{name}_bs{batch_size}_sz{img_size}_obs{outbatches}_s{ranges[0]}_e{ranges[1]}.pth"
                torch.save(feats[name], open(os.path.join(outdir, na), "wb"))
            na = f"targets_bs{batch_size}_sz{img_size}_obs{outbatches}_s{ranges[0]}_e{ranges[1]}.pth"
            torch.save(all_targets, open(os.path.join(outdir, na), "wb"))

    @staticmethod
    def merge_feats(features=[], targets=[], length=1281167, save="/tmp/1.pth"):
        feats = [torch.load(open(f, "rb")) for f in features]
        tgts = [torch.load(open(f, "rb")) for f in targets]
        for i, (f, t) in enumerate(zip(feats, tgts)):
            assert f.shape[0:2] == t.shape[0:2], breakpoint()
        assert sum([t.shape[0] for t in tgts]) * tgts[0].shape[1] >= length
        print(features, targets, flush=True)
        feats = torch.cat(feats, dim=0).view(-1, feats[0].shape[-1])
        tgts = torch.cat(tgts, dim=0).view(-1)
        if not (len(feats) == length):
            assert (feats[length:] == feats[length]).all() # input 0, models output same
            assert (feats[length] != feats[length - 1]).any()
            assert (tgts[length:] == -1).all()
            assert (tgts[:length] != -1).all()
        feats = feats[:length]
        tgts = tgts[:length]
        os.makedirs(os.path.dirname(save), exist_ok=True)
        assert not os.path.exists(save), f"file {save} exist"
        torch.save(dict(features=feats, targets=tgts), open(save, "wb"))


# used for build models
class BuildModels:
    @staticmethod
    def build_vheat(with_ckpt=False, remove_head=False, only_backbone=False, scale="small", size=224):
        assert not with_ckpt 
        assert not remove_head
        assert not only_backbone
        print("vheat ================================", flush=True)
        _model = import_abspy("vheat", f"{HOME}/packs/VHeat/classification/models")
        VHEAT = _model.HeatM_V2_Stem_Noangle_Freqembed_Oldhead_Fast2_Torelease
        tiny = partial(VHEAT, depths=[2, 2, 6, 2], dims=96, img_size=size, infer_mode=True)
        small = partial(VHEAT, depths=[2, 2, 18, 2], dims=96, img_size=size, infer_mode=True)
        base = partial(VHEAT, depths=[2, 2, 18, 2], dims=128, img_size=size, infer_mode=True)
        model = dict(tiny=tiny, small=small, base=base)[scale]()
        model.infer_init()

        return model
    
    @staticmethod
    def build_visionmamba(with_ckpt=False, remove_head=False, only_backbone=False, scale="small", size=224):
        assert not with_ckpt 
        assert not remove_head
        assert not only_backbone
        print("vim ================================", flush=True)
        specpath = f"{HOME}/packs/Vim/mamba-1p1p1"
        sys.path.insert(0, specpath)
        import mamba_ssm
        _model = import_abspy("models_mamba", f"{HOME}/packs/Vim/vim")
        model = _model.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
        sys.path = sys.path[1:]
        return model

    @staticmethod
    def build_s4nd(with_ckpt=False, remove_head=False, only_backbone=False, scale="ctiny", size=224):
        assert not with_ckpt 
        assert not remove_head
        assert scale in ["vitb", "ctiny"]
        print("convnext-s4nd ================================", flush=True)
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./convnexts4nd")
        sys.path.insert(0, specpath)
        import timm; assert timm.__version__ == "0.5.4"
        import structured_kernels
        model = import_abspy("vit_all", f"{os.path.dirname(__file__)}/convnexts4nd")
        vitb = model.vit_base_s4nd
        model = import_abspy("convnext_timm", f"{os.path.dirname(__file__)}/convnexts4nd")
        ctiny = model.convnext_tiny_s4nd
        model = dict(ctiny=ctiny, vitb=vitb)[scale]()
        sys.path = sys.path[1:]
        
        if only_backbone:
            model.forward = model.forward_features

        return model

    @staticmethod
    def build_vmamba(with_ckpt=False, remove_head=False, only_backbone=False, scale="tv0", size=224, cfg=None, ckpt=None, key="model"):
        print("vssm ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        if scale == "flex":
            model = _model.VSSM(**cfg)
            ckpt = ckpt
        else:
            tv2 = (
                partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth"
            )
            sv2 = (
                partial(_model.VSSM, dims=96, depths=[2,2,15,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_small_0229/vssm1_small_0229_ckpt_epoch_222.pth"
            )
            bv2 = (
                partial(_model.VSSM, dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_base_0229/vssm1_base_0229_ckpt_epoch_237.pth"
            )
            tv1 = (
                partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230/vssm1_tiny_0230_ckpt_epoch_262.pth"
            )
            tv0 = (
                partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
            )
            sv0 = (
                partial(_model.VSSM, dims=96, depths=[2,2,27,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmsmall/vssmsmall_dp03_ckpt_epoch_238.pth"
            )
            bv0 = (
                partial(_model.VSSM, dims=128, depths=[2,2,27,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d"),
                f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmbase/vssmbase_dp06_ckpt_epoch_241.pth"
            )
            model = dict(tv0=tv0, tv1=tv1, tv2=tv2, sv0=sv0, sv2=sv2, bv0=bv0, bv2=bv2)[scale][0]()
            ckpt = dict(tv0=tv0, tv1=tv1, tv2=tv2, sv0=sv0, sv2=sv2, bv0=bv0, bv2=bv2)[scale][1]

        if with_ckpt:
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))[key])

        if remove_head:
            print(model.classifier.head, flush=True)
            model.classifier.head = nn.Identity() # 768->1000
        elif only_backbone:
            def _forward(self, x: torch.Tensor):
                x = self.patch_embed(x)
                if self.pos_embed is not None:
                    pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
                    x = x + pos_embed
                for layer in self.layers:
                    x = layer(x)
                if not self.channel_first:
                    x = x.permute(0, 3, 1, 2).contiguous()
                return x
            model.forward = partial(_forward, model)
        return model

    @staticmethod
    def build_swin(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        print("swin ================================", flush=True)
        specpath = f"{HOME}/packs/Swin-Transformer"
        sys.path.insert(0, specpath)
        import swin_window_process
        _model = import_abspy("swin_transformer", f"{HOME}/packs/Swin-Transformer/models")
        # configs/swin/swin_tiny_patch4_window7_224.yaml
        tiny = partial(_model.SwinTransformer, embed_dim=96, depths=[2,2,6,2], num_heads=[ 3, 6, 12, 24 ], img_size=size, window_size=(size//32), fused_window_process=True)
        # configs/swin/swin_small_patch4_window7_224.yaml
        small = partial(_model.SwinTransformer, embed_dim=96, depths=[2,2,18,2], num_heads=[ 3, 6, 12, 24 ], img_size=size, window_size=(size//32), fused_window_process=True)
        # # configs/swin/swin_base_patch4_window7_224.yaml
        base = partial(_model.SwinTransformer, embed_dim=128, depths=[2,2,18,2], num_heads=[ 4, 8, 16, 32 ], img_size=size, window_size=(size//32), fused_window_process=True)
        sys.path = sys.path[1:]
        model = dict(tiny=tiny, small=small, base=base)[scale]()

        if with_ckpt:
            assert size == 224, "only support size 224"
            assert scale == "tiny", "support tiny with ckpt only"
            ckpt = f"{HOME}/packs/ckpts/swin_tiny_patch4_window7_224.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])

        if remove_head:
            print(model.head, flush=True)
            model.head = nn.Identity()
        elif only_backbone:
            def _forward(self, x):
                x = self.patch_embed(x)
                if self.ape:
                    x = x + self.absolute_pos_embed
                x = self.pos_drop(x)

                for layer in self.layers:
                    x = layer(x)
                x = x.permute(0, 2, 1)
                x = x.view(*x.shape[0:2], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1])))
                return x

            model.forward = partial(_forward, model)

        return model
      
    @staticmethod
    def build_convnext(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        print("convnext ================================", flush=True)
        _model = import_abspy("convnext", f"{HOME}/packs/ConvNeXt/models")
        tiny = _model.convnext_tiny()
        small = _model.convnext_small()
        base = _model.convnext_base()
        model = dict(tiny=tiny, small=small, base=base)[scale]

        if with_ckpt:
            assert scale == "tiny", "support tiny with ckpt only"
            ckpt =f"{HOME}/packs/ckpts/convnext_tiny_1k_224_ema.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])

        if remove_head:
            print(model.head, flush=True)
            model.head = nn.Identity() # 768
        elif only_backbone:
            def _forward(self, x):
                for i in range(4):
                    x = self.downsample_layers[i](x)
                    x = self.stages[i](x)
                return x
            model.forward = partial(_forward, model)
        return model

    @staticmethod
    def build_hivit(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        print("hivit [for testing throughput only] ================================", flush=True)
        sys.path.insert(0, "")
        _model = import_abspy("hivit", f"{HOME}/packs/hivit/supervised/models/")
        tiny = partial(_model.HiViT, img_size=size, patch_size=16, inner_patches=4, embed_dim=384, depths=[1, 1, 10], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        small = partial(_model.HiViT, img_size=size, patch_size=16, inner_patches=4, embed_dim=384, depths=[2, 2, 20], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        base = partial(_model.HiViT, img_size=size, patch_size=16, inner_patches=4, embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        sys.path = sys.path[1:]
        model = dict(tiny=tiny, small=small, base=base)[scale]()

        if with_ckpt:
            assert NotImplementedError
        if remove_head:
            assert NotImplementedError
        elif only_backbone:
           assert NotImplementedError
        return model
    
    @staticmethod
    def build_intern(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        print("intern ================================", flush=True)
        specpath = f"{HOME}/packs/InternImage/classification"
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/packs/InternImage/classification/models/")
        sys.path = sys.path[1:]
        tiny = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        small = partial(_model.InternImage, core_op='DCNv3', channels=80, depths=[4, 4, 21, 4], groups=[5, 10, 20, 40], layer_scale=1e-5, offset_scale=1.0, mlp_ratio=4., post_norm=True)
        base = partial(_model.InternImage, core_op='DCNv3', channels=112, depths=[4, 4, 21, 4], groups=[7, 14, 28, 56], layer_scale=1e-5, offset_scale=1.0, mlp_ratio=4., post_norm=True)
        model = dict(tiny=tiny, small=small, base=base)[scale]()
        
        if with_ckpt:
            assert scale == "tiny", "only support tiny model"
            ckpt = f"{HOME}/packs/ckpts/internimage_t_1k_224.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        
        if remove_head:
            print(model.head, flush=True) # 768
            model.head = nn.Identity()
        elif only_backbone:
            def forward(self, x):
                x = self.patch_embed(x)
                x = self.pos_drop(x)

                for level in self.levels:
                    x = level(x)
                return x.permute(0, 3, 1, 2)
            
            model.forward = partial(forward, model)
        return model

    @staticmethod
    def build_xcit(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        print("xcit =================", flush=True)
        xcit = import_abspy("xcit", f"{HOME}/packs/xcit/")
        model = dict(tiny=xcit.xcit_small_12_p16, small=xcit.xcit_small_24_p16, base=xcit.xcit_medium_24_p16)[scale]()

        if with_ckpt:
            assert scale == "tiny", "only support tiny for ckpt"
            ckpt = f"{HOME}/packs/ckpts/xcit_small_12_p16_224.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])

        if remove_head:
            print(model.head, flush=True)
            def forward(self, x):
                x = self.forward_features(x)
                return x
            model.forward = partial(forward, model)
        elif only_backbone:
            def _forward(self, x):
                B, C, H, W = x.shape

                x, (Hp, Wp) = self.patch_embed(x)

                if self.use_pos:
                    pos_encoding = self.pos_embeder(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
                    x = x + pos_encoding

                x = self.pos_drop(x)

                for blk in self.blocks:
                    x = blk(x, Hp, Wp)

                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                for blk in self.cls_attn_blocks:
                    x = blk(x, Hp, Wp)

                x = x[:, 1:, :].permute(0, 2, 1)
                x = x.view(*x.shape[0:2], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1])),)
                return x
            model.forward = partial(_forward, model)
        else:
            def forward(self, x):
                x = self.forward_features(x)
                return x
            model.forward = partial(forward, model)
        return model

    @staticmethod
    def build_swin_mmpretrain(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        print("swin scale [do not test throughput with this]================================", flush=True)
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier

        model = dict(
            type='ImageClassifier',
            backbone=dict(
                type='SwinTransformer', arch=scale, img_size=224, drop_path_rate=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=1024 if scale == "base" else 768,
                init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
        )
        ckpt = "https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth"
        model["backbone"].update({"window_size": int(size // 32)})
        model: ImageClassifier = build_classifier(model)
        if with_ckpt:
            assert scale == "tiny", "support tiny with ckpt only"
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)

        if remove_head:
            print(model.head.fc, flush=True) # 768
            model.head.fc = nn.Identity()
        elif only_backbone:
            def forward_backbone(self: ImageClassifier, x):
                x = self.backbone(x)[-1]
                return x
            model.forward = partial(forward_backbone, model)
        return model
      
    @staticmethod
    def build_hivit_mmpretrain(with_ckpt=False, remove_head=False, only_backbone=False, scale="tiny", size=224):
        assert scale == "tiny", "support tiny only"
        print("hivit scale [do not test throughput with this]================================", flush=True)
        from mmpretrain.models.builder import MODELS
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, HiViT, SwinTransformer
        from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np
        
        class _HiViTx(HiViT):
            def __init__(self, *args,**kwargs):
                super().__init__(*args,**kwargs)
                self.num_extra_tokens = 0
                self.interpolate_mode = "bicubic"
                self.patch_embed.init_out_size = self.patch_embed.patches_resolution
                self._register_load_state_dict_pre_hook(self._prepare_abs_pos_embed)
                self._register_load_state_dict_pre_hook(
                    self._prepare_relative_position_bias_table)

            # copied from SwinTransformer, change absolute_pos_embed to pos_embed
            def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
                name = prefix + 'pos_embed'
                if name not in state_dict.keys():
                    return

                ckpt_pos_embed_shape = state_dict[name].shape
                if self.pos_embed.shape != ckpt_pos_embed_shape:
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(
                        'Resize the pos_embed shape from '
                        f'{ckpt_pos_embed_shape} to {self.pos_embed.shape}.')

                    ckpt_pos_embed_shape = to_2tuple(
                        int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
                    pos_embed_shape = self.patch_embed.init_out_size

                    state_dict[name] = resize_pos_embed(state_dict[name],
                                                        ckpt_pos_embed_shape,
                                                        pos_embed_shape,
                                                        self.interpolate_mode,
                                                        self.num_extra_tokens)

            def _prepare_relative_position_bias_table(self, state_dict, *args, **kwargs):
                del state_dict['backbone.relative_position_index']
                aaa = SwinTransformer._prepare_relative_position_bias_table(self, state_dict, *args, **kwargs)
                return aaa

        try:
            @MODELS.register_module()
            class HiViTx(_HiViTx):
                ...
        except Exception as e:
            print(e)

        print("hivit ================================", flush=True)
        model = dict(
            backbone=dict(
                ape=True,
                arch='tiny',
                drop_path_rate=0.05,
                img_size=224,
                rpe=True,
                type='HiViTx'),
            head=dict(
                cal_acc=False,
                in_channels=384,
                init_cfg=None,
                loss=dict(
                    label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
                num_classes=1000,
                type='LinearClsHead'),
            init_cfg=[
                dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
                dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
            ],
            neck=dict(type='GlobalAveragePooling'),
            train_cfg=dict(augments=[
                dict(alpha=0.8, type='Mixup'),
                dict(alpha=1.0, type='CutMix'),
            ]),
            type='ImageClassifier')
        model["backbone"].update({"img_size": size})
        model = build_classifier(model)

        if with_ckpt:
            assert scale == "tiny", "support tiny with ckpt only"
            ckpt = f"{HOME}/packs/ckpts/hivit-tiny-p16_8xb128_in1k/epoch_295.pth"
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)

        if remove_head:
            print(model.head.fc, flush=True) # 768
            model.head.fc = nn.Identity()
        elif only_backbone:
            def forward_backbone(self: ImageClassifier, x):
                x = self.backbone(x)[-1]
                return x
            model.forward = partial(forward_backbone, model)
        return model

    @staticmethod
    def build_deit_mmpretrain(with_ckpt=False, remove_head=False, only_backbone=False, scale="small", size=224, test_flops=False):
        print("deit ================================", flush=True)
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, HiViT, VisionTransformer, SwinTransformer
        from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np
        
        small = dict(
            type='ImageClassifier',
            backbone=dict(
                type='VisionTransformer',
                arch='deit-small',
                img_size=size,
                patch_size=16),
            neck=None,
            head=dict(
                type='VisionTransformerClsHead',
                num_classes=1000,
                in_channels=384,
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
            ),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=.02),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
        )
        base = dict(
            type='ImageClassifier',
            backbone=dict(
                type='VisionTransformer',
                arch='deit-base',
                img_size=size,
                patch_size=16,
                drop_path_rate=0.1),
            neck=None,
            head=dict(
                type='VisionTransformerClsHead',
                num_classes=1000,
                in_channels=768,
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
            ),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=.02),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
        )

        model = dict(small=small, base=base)[scale]
        ckpt = dict(
            small="https://download.openmmlab.com/mmclassification/v0/deit/deit-small_pt-4xb256_in1k_20220218-9425b9bb.pth",
            base="https://download.openmmlab.com/mmclassification/v0/deit/deit-base_pt-16xb64_in1k_20220216-db63c16c.pth",
        )[scale]

        model = build_classifier(model)

        if with_ckpt:
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        if remove_head:
            print(model.head.layers.head, flush=True)
            model.head.layers.head = nn.Identity() # 384->1000
        elif only_backbone:
            model.backbone.out_type = 'featmap'
            def forward_backbone(self: ImageClassifier, x):
                x = self.backbone(x)[-1]
                return x
            model.forward = partial(forward_backbone, model)
        
        if test_flops:
            print("WARNING: this mode may make throughput lower, used to test flops only!", flush=True)
            from mmpretrain.models.utils.attention import scaled_dot_product_attention_pyimpl
            for layer in model.backbone.layers:
                layer.attn.scaled_dot_product_attention = scaled_dot_product_attention_pyimpl
        else:
            print("WARNING: this mode will make flops lower, do not use this to test flops!", flush=True)
        return model

    @staticmethod
    def build_resnet_mmpretrain(with_ckpt=False, remove_head=False, only_backbone=False, scale="r50", size=224):
        print("resnet ================================", flush=True)
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier
        
        r50 = dict(
            type='ImageClassifier',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3, ),
                style='pytorch'),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=2048,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5),
            ))

        r101 = dict(
            type='ImageClassifier',
            backbone=dict(
                type='ResNet',
                depth=101,
                num_stages=4,
                out_indices=(3, ),
                style='pytorch'),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=2048,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5),
            ))

        model = dict(r50=r50, r101=r101)[scale]
        ckpt = dict(
            r50="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth",
            r101="https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth",
        )[scale]

        model = build_classifier(model)

        if with_ckpt:
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'])
        if remove_head:
            print(model.head.fc, flush=True)
            model.head.fc = nn.Identity() # 2048->1000
        elif only_backbone:
            def forward_backbone(self: ImageClassifier, x):
                x = self.backbone(x)[-1]
                return x
            model.forward = partial(forward_backbone, model)
        return model

    @staticmethod
    def build_replknet31b_mmpretrain(with_ckpt=False, remove_head=False, only_backbone=False, scale="31b", size=224):
        print("replknet31b ================================", flush=True)
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier
        
        replknet31b = dict(
            type='ImageClassifier',
            backbone=dict(
                type='RepLKNet',
                arch='31B',
                out_indices=(3, ),
            ),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=1024,
                loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5),
            ))

        ckpt = "https://download.openmmlab.com/mmclassification/v0/replknet/replknet-31B_3rdparty_in1k_20221118-fd08e268.pth"
        model = build_classifier(replknet31b)

        if with_ckpt:
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'])
        if remove_head:
            print(model.head.fc, flush=True)
            model.head.fc = nn.Identity()
        elif only_backbone:
            def forward_backbone(self: ImageClassifier, x):
                x = self.backbone(x)[-1]
                return x
            model.forward = partial(forward_backbone, model)
        return model

    @staticmethod
    def build_mmpretrain_models(cfg="swin_tiny", ckpt=True, only_backbone=False, with_norm=True, **kwargs):
        import os
        from functools import partial
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
        from mmengine.config import Config
        config_root = os.path.join(os.path.dirname(__file__), "../../analyze/mmpretrain_configs/configs/") 
        
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

        if cfg not in CFGS:
            return None

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

        return model

    @classmethod
    def check(cls):
        for mbuild in [
            # partial(cls.build_vmamba, scale="tv0"),
            # partial(cls.build_vmamba, scale="tv1"), 
            # partial(cls.build_vmamba, scale="tv2"),
            # partial(cls.build_vmamba, scale="sv0"),
            # partial(cls.build_vmamba, scale="sv2"),
            # partial(cls.build_vmamba, scale="bv0"),
            # partial(cls.build_vmamba, scale="bv2"),
            # partial(cls.build_swin, scale="tiny"),
            # partial(cls.build_swin, scale="small"),
            # partial(cls.build_swin, scale="base"),
            # partial(cls.build_convnext, scale="tiny"),
            # partial(cls.build_convnext, scale="small"),
            # partial(cls.build_convnext, scale="base"),
            # partial(cls.build_hivit, scale="tiny"),
            # partial(cls.build_hivit, scale="small"),
            # partial(cls.build_hivit, scale="base"), 
            # partial(cls.build_intern, scale="tiny"), 
            # partial(cls.build_intern, scale="small"), 
            # partial(cls.build_intern, scale="base"), 
            # partial(cls.build_xcit, scale="tiny"), 
            # partial(cls.build_xcit, scale="small"), 
            # partial(cls.build_xcit, scale="base"), 
            # partial(cls.build_swin_mmpretrain, scale="tiny"), 
            # partial(cls.build_swin_mmpretrain, scale="small"), 
            # partial(cls.build_swin_mmpretrain, scale="base"), 
            # partial(cls.build_hivit_mmpretrain, scale="tiny"), 
            # partial(cls.build_hivit_mmpretrain, scale="small"), 
            # partial(cls.build_hivit_mmpretrain, scale="base"), 
            # partial(cls.build_deit_mmpretrain, scale="small"), 
            # partial(cls.build_deit_mmpretrain, scale="base"), 
            # partial(cls.build_resnet_mmpretrain, scale="r50"), 
            # partial(cls.build_resnet_mmpretrain, scale="r101"), 
            # partial(cls.build_replknet31b_mmpretrain, scale="31b"), 
        ]:
            for size in [224, 768]:
                inp = torch.randn((2, 3, size, size)).cuda()
                for with_ckpt in [False, True]:
                    for remove_head in [False, True]:
                        for only_backbone in [False, True]:
                            if False:
                                model = mbuild(with_ckpt=with_ckpt, remove_head=remove_head, only_backbone=only_backbone, size=size).cuda()
                                print(size, with_ckpt, remove_head, only_backbone, model(inp).shape, flush=True)
                            try:
                                model = mbuild(with_ckpt=with_ckpt, remove_head=remove_head, only_backbone=only_backbone).cuda()
                                print(size, with_ckpt, remove_head, only_backbone, model(inp).shape, flush=True)
                            except Exception as e:
                                print(size, with_ckpt, remove_head, only_backbone, flush=True)
                                print("ERROR:", e, flush=True)
        breakpoint()


# used for print flops
class FLOPs:
    @staticmethod
    def register_supported_ops():
        build = import_abspy("models", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"))
        selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit
        # flops_selective_scan_fn: Callable = build.vmamba.flops_selective_scan_fn
        # flops_selective_scan_ref: Callable = build.vmamba.flops_selective_scan_ref 
        def causal_conv_1d_jit(inputs, outputs):
            """
            https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
            x: (batch, dim, seqlen) weight: (dim, width) bias: (dim,) out: (batch, dim, seqlen)
            out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
            """
            from fvcore.nn.jit_handles import conv_flop_jit
            return conv_flop_jit(inputs, outputs)
        
        supported_ops={
            "aten::gelu": None, # as relu is in _IGNORED_OPS
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # =====================================================
            # for mamba-ssm
            "prim::PythonOp.CausalConv1dFn": causal_conv_1d_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
            # =====================================================
            # for VMamba
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            # "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,
            # =====================================================
            # "aten::scaled_dot_product_attention": ...
        }
        return supported_ops

    @staticmethod
    def check_operations(model: nn.Module, inputs=None, input_shape=(3, 224, 224)):
        from fvcore.nn.jit_analysis import _get_scoped_trace_graph, _named_modules_with_dup, Counter, JitModelAnalysis
        
        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 4

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)


        model.eval()

        flop_counter = JitModelAnalysis(model, inputs)
        flop_counter._ignored_ops = set()
        flop_counter._op_handles = dict()
        assert flop_counter.total() == 0 # make sure no operations supported
        print(flop_counter.unsupported_ops(), flush=True)
        print(f"supported ops {flop_counter._op_handles}; ignore ops {flop_counter._ignored_ops};", flush=True)

    @classmethod
    def fvcore_flop_count(cls, model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False, show_arch=False, verbose=True):
        supported_ops = cls.register_supported_ops()
        from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
        from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
        from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
        from fvcore.nn.jit_analysis import _IGNORED_OPS
        from fvcore.nn.jit_handles import get_shape, addmm_flop_jit
        
        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 4

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)


        model.eval()

        Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)
        
        flops_table = flop_count_table(
            flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
            max_depth=100,
            activations=None,
            show_param_shapes=True,
        )

        flops_str = flop_count_str(
            flops = FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
            activations=None,
        )

        if show_arch:
            print(flops_str)

        if show_table:
            print(flops_table)

        params = fvcore_parameter_count(model)[""]
        flops = sum(Gflops.values())

        if verbose:
            print(Gflops.items())
            print("GFlops: ", flops, "Params: ", params, flush=True)
        
        return params, flops

    # equals with fvcore_flop_count
    @classmethod
    def mmengine_flop_count(cls, model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
        supported_ops = cls.register_supported_ops()
        from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
        from mmengine.analysis.jit_analysis import _IGNORED_OPS
        from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
        from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info
        
        # modified from mmengine.analysis
        def get_model_complexity_info(
            model: nn.Module,
            input_shape: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...],
                            None] = None,
            inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Tuple[Any, ...],
                        None] = None,
            show_table: bool = True,
            show_arch: bool = True,
        ):
            if input_shape is None and inputs is None:
                raise ValueError('One of "input_shape" and "inputs" should be set.')
            elif input_shape is not None and inputs is not None:
                raise ValueError('"input_shape" and "inputs" cannot be both set.')

            if inputs is None:
                device = next(model.parameters()).device
                if is_tuple_of(input_shape, int):  # tuple of int, construct one tensor
                    inputs = (torch.randn(1, *input_shape).to(device), )
                elif is_tuple_of(input_shape, tuple) and all([
                        is_tuple_of(one_input_shape, int)
                        for one_input_shape in input_shape  # type: ignore
                ]):  # tuple of tuple of int, construct multiple tensors
                    inputs = tuple([
                        torch.randn(1, *one_input_shape).to(device)
                        for one_input_shape in input_shape  # type: ignore
                    ])
                else:
                    raise ValueError(
                        '"input_shape" should be either a `tuple of int` (to construct'
                        'one input tensor) or a `tuple of tuple of int` (to construct'
                        'multiple input tensors).')

            flop_handler = FlopAnalyzer(model, inputs).set_op_handle(**supported_ops)
            # activation_handler = ActivationAnalyzer(model, inputs)

            flops = flop_handler.total()
            # activations = activation_handler.total()
            params = parameter_count(model)['']

            flops_str = _format_size(flops)
            # activations_str = _format_size(activations)
            params_str = _format_size(params)

            if show_table:
                complexity_table = complexity_stats_table(
                    flops=flop_handler,
                    # activations=activation_handler,
                    show_param_shapes=True,
                )
                complexity_table = '\n' + complexity_table
            else:
                complexity_table = ''

            if show_arch:
                complexity_arch = complexity_stats_str(
                    flops=flop_handler,
                    # activations=activation_handler,
                )
                complexity_arch = '\n' + complexity_arch
            else:
                complexity_arch = ''

            return {
                'flops': flops,
                'flops_str': flops_str,
                # 'activations': activations,
                # 'activations_str': activations_str,
                'params': params,
                'params_str': params_str,
                'out_table': complexity_table,
                'out_arch': complexity_arch
            }
        
        if _get_model_complexity_info:
            return get_model_complexity_info

        model.eval()
        analysis_results = get_model_complexity_info(
            model,
            input_shape,
            show_table=show_table,
            show_arch=show_arch,
        )
        flops = analysis_results['flops_str']
        params = analysis_results['params_str']
        # activations = analysis_results['activations_str']
        out_table = analysis_results['out_table']
        out_arch = analysis_results['out_arch']
        
        if show_arch:
            print(out_arch)
        
        if show_table:
            print(out_table)
        
        split_line = '=' * 30
        print(f'{split_line}\nInput shape: {input_shape}\t'
            f'Flops: {flops}\tParams: {params}\t'
            #   f'Activation: {activations}\n{split_line}'
        , flush=True)
        # print('!!!Only the backbone network is counted in FLOPs analysis.')
        # print('!!!Please be cautious if you use the results in papers. '
        #       'You may need to check if all ops are supported and verify that the '
        #       'flops computation is correct.')

    @classmethod
    def mmdet_flops(cls, config=None, extra_config=None):
        from mmengine.config import Config
        from mmengine.runner import Runner
        import numpy as np
        import os

        cfg = Config.fromfile(config)
        if "model" in cfg:
            if "pretrained" in cfg["model"]:
                cfg["model"].pop("pretrained")
        if extra_config is not None:
            new_cfg = Config.fromfile(extra_config)
            new_cfg["model"] = cfg["model"]
            cfg = new_cfg
        cfg["work_dir"] = "/tmp"
        cfg["default_scope"] = "mmdet"
        runner = Runner.from_cfg(cfg)
        model = runner.model.cuda()
        get_model_complexity_info = cls.mmengine_flop_count(_get_model_complexity_info=True)
        
        if True:
            oridir = os.getcwd()
            os.chdir(os.path.join(os.path.dirname(__file__), "../detection"))
            data_loader = runner.val_dataloader
            num_images = 100
            mean_flops = []
            for idx, data_batch in enumerate(data_loader):
                if idx == num_images:
                    break
                data = model.data_preprocessor(data_batch)
                model.forward = partial(model.forward, data_samples=data['data_samples'])
                # out = get_model_complexity_info(model, inputs=data['inputs'])
                out = get_model_complexity_info(model, input_shape=(3, 1280, 800))
                params = out['params_str']
                mean_flops.append(out['flops'])
            mean_flops = np.average(np.array(mean_flops))
            print(params, mean_flops)
            os.chdir(oridir)

    @classmethod
    def mmseg_flops(cls, config=None, input_shape=(3, 512, 2048)):
        from mmengine.config import Config
        from mmengine.runner import Runner

        cfg = Config.fromfile(config)
        cfg["work_dir"] = "/tmp"
        cfg["default_scope"] = "mmseg"
        runner = Runner.from_cfg(cfg)
        model = runner.model.cuda()
        
        cls.fvcore_flop_count(model, input_shape=input_shape)


if __name__ == "__main__":
    BuildModels.check()
