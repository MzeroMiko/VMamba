import torch.nn as nn
import time
import seaborn

import os
import torch
from functools import partial
import numpy as np

# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def analyze_erf(source="/tmp/erf.npy", dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    # A script to visualize the ERF.
    # Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
    # Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
    # Licensed under The MIT License [see LICENSE for details]
    # --------------------------------------------------------'
    import argparse
    import matplotlib.pyplot as plt
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


    # parser = argparse.ArgumentParser('Script for analyzing the ERF', add_help=False)
    # parser.add_argument('--source', default='temp.npy', type=str, help='path to the contribution score matrix (.npy file)')
    # parser.add_argument('--heatmap_save', default='heatmap.png', type=str, help='where to save the heatmap')
    # args = parser.parse_args()

    class Args():
        ...
    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM


    import numpy as np

    def heatmap(data, camp='RdYlGn', figsize=(10, 10.75), ax=None, save_path=None):
        plt.figure(figsize=figsize, dpi=40)

        ax = sns.heatmap(data,
                    xticklabels=False,
                    yticklabels=False, cmap=camp,
                    center=0, annot=False, ax=ax, cbar=True, annot_kws={"size": 24}, fmt='.2f')
        #   =========================== Add a **nicer** colorbar on top of the figure. Works for matplotlib 3.3. For later versions, use matplotlib.colorbar
        #   =========================== or you may simply ignore these and set cbar=True in the heatmap function above.
        # from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        # from mpl_toolkits.axes_grid1.colorbar import colorbar

        # ax_divider = make_axes_locatable(ax)
        # cax = ax_divider.append_axes('top', size='5%', pad='2%')
        # colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
        # cax.xaxis.set_ticks_position('top')
        #   ================================================================
        #   ================================================================
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


    # if __name__ == '__main__':
    if True:
        import os
        os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
        analyze_erf(args)


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module=None, num_images=50, data_path="/dataset/ImageNet2012", save_path=f"/tmp/{time.time()}/erf.npy"):
    # A script to visualize the ERF.
    # Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
    # Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
    # Licensed under The MIT License [see LICENSE for details]
    # --------------------------------------------------------'

    import os
    import argparse
    import numpy as np
    import torch
    from timm.utils import AverageMeter
    from torchvision import datasets, transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from PIL import Image
    # from resnet_for_erf import resnet101, resnet152
    # from replknet_for_erf import RepLKNetForERF
    from torch import optim as optim


    # def parse_args():
    #     parser = argparse.ArgumentParser('Script for visualizing the ERF', add_help=False)
    #     parser.add_argument('--model', default='resnet101', type=str, help='model name')
    #     parser.add_argument('--weights', default=None, type=str, help='path to weights file. For resnet101/152, ignore this arg to download from torchvision')
    #     parser.add_argument('--data_path', default='path_to_imagenet', type=str, help='dataset path')
    #     parser.add_argument('--save_path', default='temp.npy', type=str, help='path to save the ERF matrix (.npy file)')
    #     parser.add_argument('--num_images', default=50, type=int, help='num of images to use')
    #     args = parser.parse_args()
    #     return args


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
        # nori_root = os.path.join('/home/dingxiaohan/ndp/', 'imagenet.val.nori.list')
        # from nori_dataset import ImageNetNoriDataset      # Data source on our machines. You will never need it.
        # dataset = ImageNetNoriDataset(nori_root, transform=transform)

        from torch.utils.data import SequentialSampler, DataLoader, RandomSampler
        sampler_val = RandomSampler(dataset)
        data_loader_val = DataLoader(dataset, sampler=sampler_val,
            batch_size=1, num_workers=1, pin_memory=True, drop_last=False)

        # if args.model == 'resnet101':
        #     model = resnet101(pretrained=args.weights is None)
        # elif args.model == 'resnet152':
        #     model = resnet152(pretrained=args.weights is None)
        # elif args.model == 'RepLKNet-31B':
        #     model = RepLKNetForERF(large_kernel_sizes=[31,29,27,13], layers=[2,2,18,2], channels=[128,256,512,1024],
        #                 small_kernel=5, small_kernel_merged=False)
        # elif args.model == 'RepLKNet-13':
        #     model = RepLKNetForERF(large_kernel_sizes=[13] * 4, layers=[2,2,18,2], channels=[128,256,512,1024],
        #                 small_kernel=5, small_kernel_merged=False)
        # else:
        #     raise ValueError('Unsupported model. Please add it here.')

        # if args.weights is not None:
        #     print('load weights')
        #     weights = torch.load(args.weights, map_location='cpu')
        #     if 'model' in weights:
        #         weights = weights['model']
        #     if 'state_dict' in weights:
        #         weights = weights['state_dict']
        #     model.load_state_dict(weights)
        #     print('loaded')

        model = MODEL
        model.cuda()
        model.eval()    #   fix BN and droppath

        optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()

        for _, (samples, _) in enumerate(data_loader_val):

            if meter.count == args.num_images:
                np.save(args.save_path, meter.avg)
                # exit()
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


    # if __name__ == '__main__':
    #     args = parse_args()
    #     main(args)
    if True:
        class Args():
            ...
        args = Args()
        args.num_images = num_images
        args.data_path = data_path
        args.save_path = save_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        main(args, MODEL)



if __name__ == "__main__":
    from get_scaleup import build_vssm, build_mmpretrain_models
    
    def init_vssm(ckpt=False, **kwargs):
        _ckpt="/home/LiuYue/Workspace3/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth"
        if ckpt:
            ckpt = _ckpt
        return build_vssm(ckpt=ckpt, only_backbone=True, with_norm=True, depths=[2, 2, 9, 2], dims=96,)
    
    init_swin = partial(build_mmpretrain_models, cfg="swin_tiny", ckpt=False, only_backbone=True, with_norm=True,)
    init_convnext = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=False, only_backbone=True, with_norm=True,)
    init_replknet = partial(build_mmpretrain_models, cfg="replknet_base", ckpt=False, only_backbone=True, with_norm=True,)
    init_deit = partial(build_mmpretrain_models, cfg="deit_small", ckpt=False, only_backbone=True, with_norm=True,)
    init_resnet50 = partial(build_mmpretrain_models, cfg="resnet50", ckpt=False, only_backbone=True, with_norm=True,)

    for model in ["vssm", "swin", "convnext", "replknet", "deit", "resnet50"]:
        save_path = f"/tmp/{time.time()}/erf.npy"
        visualize_erf(eval(f"init_{model}(ckpt=False)"), save_path=save_path)
        analyze_erf(source=save_path, dest=f"./show/erf/heatmap_{model}_before.png")

        save_path = f"/tmp/{time.time()}/erf.npy"
        visualize_erf(eval(f"init_{model}(ckpt=True)"), save_path=save_path)
        analyze_erf(source=save_path, dest=f"./show/erf/heatmap_{model}_after.png")


# CUDA_VISIBLE_DEVICES=6 python analyze/get_erf.py > ./show/erf/get_erf.log 2>&1
