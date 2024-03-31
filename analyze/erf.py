import os
import time
from functools import partial
from typing import Callable

import seaborn
import numpy as np

import torch
import torch.nn as nn
from torch import optim as optim
from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
from torchvision import datasets, transforms

from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image

if True:
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns
    #   Set figure parameters
    large = 24; med = 24; small = 24
    params = {'axes.titlesize': large,
            'legend.fontsize': med,
            'figure.figsize': (16, 10),
            'axes.labelsize': med,
            'xtick.labelsize': med,
            'ytick.labelsize': med,
            'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white")
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False

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
build_heat_models: Callable = build.build_heat_models_


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def analyze_erf(source="/tmp/erf.npy", dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    def heatmap(data, camp='RdYlGn', figsize=(10, 10.75), ax=None, save_path=None):
        plt.figure(figsize=figsize, dpi=40)
        ax = sns.heatmap(data,
                    xticklabels=False,
                    yticklabels=False, cmap=camp,
                    center=0, annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f')
        plt.savefig(save_path)

    def get_rectangle(data, thresh):
        h, w = data.shape
        all_sum = np.sum(data)
        for i in range(1, h // 2):
            selected_area = data[h // 2 - i:h // 2 + 1 + i, w // 2 - i:w // 2 + 1 + i]
            area_sum = np.sum(selected_area)
            if area_sum / all_sum > thresh:
                return i * 2 + 1, (i * 2 + 1) / h * (i * 2 + 1) / w
        return None

    def analyze_erf(args):
        data = np.load(args.source)
        print(np.max(data))
        print(np.min(data))
        data = args.ALGRITHOM(data + 1)       #   the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)      #   rescale to [0,1] for the comparability among models
        print('======================= the high-contribution area ratio =====================')
        for thresh in [0.2, 0.3, 0.5, 0.99]:
            side_length, area_ratio = get_rectangle(data, thresh)
            print('thresh, rectangle side length, area ratio: ', thresh, side_length, area_ratio)
        heatmap(data, save_path=args.heatmap_save)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...
    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf(args)


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module=None, num_images=50, data_path="/dataset/ImageNet2012", save_path=f"/tmp/{time.time()}/erf.npy"):
    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()
        central_point = torch.nn.functional.relu(outputs[:, :, out_size[2] // 2, out_size[3] // 2]).sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def main(args, MODEL: nn.Module = None):
        #   ================================= transform: resize to 1024x1024
        t = [
            transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        ]
        transform = transforms.Compose(t)

        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

        from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
        sampler_val = RandomSampler(dataset)
        data_loader_val = DataLoader(dataset, sampler=sampler_val,
            batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

        model = MODEL
        model.cuda().eval()

        optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()

        for _, (samples, _) in enumerate(data_loader_val):

            if meter.count == args.num_images:
                np.save(args.save_path, meter.avg)
                return

            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True
            optimizer.zero_grad()
            contribution_scores = get_input_grad(model, samples)

            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print('accumulate', end="")
                meter.update(contribution_scores)

    class Args():
        ...
    args = Args()
    args.num_images = num_images
    args.data_path = data_path
    args.save_path = save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    main(args, MODEL)


def build_models(**kwargs):
    model = None
    if model is None:
        model = build_mmpretrain_models(**kwargs)
    if model is None:
        model = build_vssm_models(**kwargs)
    if model is None:
        model = build_heat_models(**kwargs)
    return model


NAMES = dict(
    tiny=dict(
        heat="heat_tiny",
        vssm="vssm_tiny",
        swin="swin_tiny",
        convnext="convnext_tiny",
        deit="deit_small",
        resnet="resnet50",
    ),
    small=dict(
        heat="heat_small",
        vssm="vssm_small",
        swin="swin_small",
        convnext="convnext_small",
        resnet="resnet101",
    ),
    base=dict(
        heat="heat_base",
        vssm="vssm_base",
        swin="swin_base",
        convnext="convnext_base",
        deit="deit_base",
        replknet="replknet_base",
    ),
)


def main():
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./show/erf")
    kwargs = dict(only_backbone=True, with_norm=False)
    for model in ["heat", "vssm", "swin", "convnext", "replknet", "deit", "resnet50"]:
        try:
            cfg=NAMES["tiny"][model]
        except:
            continue
        init_model = partial(build_models, cfg=NAMES["tiny"][model], **kwargs)
        save_path = f"/tmp/{time.time()}/erf.npy"
        visualize_erf(init_model(ckpt=False), save_path=save_path)
        analyze_erf(source=save_path, dest=f"{showpath}/heatmap_{model}_before.png")

        save_path = f"/tmp/{time.time()}/erf.npy"
        visualize_erf(init_model(ckpt=True), save_path=save_path)
        analyze_erf(source=save_path, dest=f"{showpath}/heatmap_{model}_after.png")


if __name__ == "__main__":
    main()
    