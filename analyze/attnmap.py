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


from utils import visualize, get_dataset, AttnMamba, import_abspy, show_mask_on_image
visualize_attnmap = visualize.visualize_attnmap
visualize_attnmaps = visualize.visualize_attnmaps
attnmap_mamba = AttnMamba.attnmap_mamba


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



def denormalize(image: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)):
    if len(image.shape) == 2:
        image = (image.cpu() * 255).to(torch.uint8).numpy()
    elif len(image.shape) == 3:
        C, H, W = image.shape
        image = image.cpu() * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        image = (image.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    image = Image.fromarray(image)
    return image


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


def visual_mamba2(As, Bs, Cs, Ds, us, dts, delta_bias, with_ws=False, with_dt=False, only_ws=False, ratio=1, tag="bcs", H=56, W=56, front_point=(0.5, 0.5), front_back=(0.7, 0.8), showpath=os.path.join(this_path, "show"), imgori=None):
    kwargs = dict(with_ws=with_ws, with_dt=with_dt, only_ws=only_ws, ratio=ratio)

    results = [(imgori, "imgori")] if imgori is not None else []
    for ret in ["a0", "a1", "a2", "a3", "all", "nall"]:
        results.append((attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, **kwargs, ret=ret, absnorm=2)[15*28+15].view(H, W), f"pos15_15_activate_{ret}"))
        if ret in ["nall"]:
            results.append((torch.diag(attnmap_mamba(As, Bs, Cs, Ds, us, dts, delta_bias, bidx=0, **kwargs, ret=ret, absnorm=2)).view(H, W), f"dias_{ret}"))
    visualize_attnmaps(results, f"{showpath}/1.jpg")


def main_vssm():
    dataset = get_dataset(root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', img_size=512, crop=False)
    dataset = get_dataset(root='/media/Disk1/Dataset/MSCOCO2014/images/', img_size=512, ret="val2014", crop=False)
    # dataset = get_dataset(root='/media/Disk1/Dataset/ADEChallengeData2016/images/', img_size=448, ret="validation", crop=False)    
    
    vssm: nn.Module = VSSM(
        depths=[2, 2, 8, 2], 
        dims=[96, 192, 384, 768], 
        ssm_d_state=1,
        ssm_ratio=1.0,
        ssm_dt_rank="auto",
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz",
        mlp_ratio=4.0,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
    ).cuda().eval()
    # vssm.load_state_dict(torch.load(open("/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth", "rb"), map_location="cpu")["model"], strict=False)
    vssm.load_state_dict(AttnMamba.convert_state_dict_from_mmdet(torch.load(open("/home/LiuYue/Workspace/PylanceAware/ckpts/private/detection/vssm1/detection/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s/epoch_36.pth", "rb"), map_location="cpu")["state_dict"]), strict=False)
    
    ss2ds = []
    for layer in vssm.layers:
        _ss2ds = []
        for blk in layer.blocks:
            ss2d = blk.op
            setattr(ss2d, "__DEBUG__", True)
            _ss2ds.append(ss2d)
        ss2ds.append(_ss2ds)
    
    showpath = os.path.join(this_path, "show/vssm")

    featHW = 32
    idxs_posxs_posys = [
        [0, 0.7, 0.3], [0, 0.2, 0.8], 
        [149, 0.7, 0.5], [149, 0.2, 0.4], 
        [162, 0.7, 0.4], [162, 0.4, 0.4], 
        [204, 0.3, 0.6], [204, 0.7, 0.2], 
        [273, 0.2, 0.6], [273, 0.9, 0.5],
        [309, 0.1, 0.7], [309, 0.9, 0.8],

    ]
    for idx, posx, posy in idxs_posxs_posys:
        img, label = dataset[idx]

        with torch.no_grad():
            out = vssm(img[None].cuda())
        print(out.argmax().item(), label, img.shape)
        os.makedirs(f"{showpath}/{idx}_{posx}_{posy}", exist_ok=True)
        denormalize(img).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()

        for m0 in ["a0", "a1", "a2", "a3", "all", "nall"]:
            for m1 in ["CB", "CwBw", "ww"]:
                aaa = AttnMamba.get_attnmap_mamba(ss2ds, 2, f"{m0}_norm_{m1}", raw_attn=True, block_id=1)
                mask = aaa[int(posy * featHW) * int(featHW) + int(posx * featHW)].view(featHW, featHW)
                visualize_attnmap(mask, f"{showpath}/{idx}_{posx}_{posy}/{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)
        # breakpoint()
    breakpoint()


def main_deit(det_model=False):
    dataset = get_dataset(root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', img_size=512, crop=False)
    dataset = get_dataset(root='/media/Disk1/Dataset/MSCOCO2014/images/', img_size=512, ret="val2014", crop=False)
    # dataset = get_dataset(root='/media/Disk1/Dataset/ADEChallengeData2016/images/', img_size=448, ret="validation", crop=False)    
    
    attns = dict()
    deit_small_baseline = None
    if det_model:
        from vit_adpter_baseline import deit_small_baseline, Attention, WindowedAttention
        sd = torch.load("/home/LiuYue/Workspace/PylanceAware/ckpts/others/deit_small_patch16_224-cd65a155.pth", map_location=torch.device("cpu"))
        deit_small_baseline = deit_small_baseline().cuda()
        deit_small_baseline.load_state_dict(sd['model'], strict=False)
        
        def attn_forward(self: Attention, x, H, W):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            setattr(self, "__data__", attn)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        
        for n, m in deit_small_baseline.blocks.named_children():
            if isinstance(m.attn, Attention):
                print(n, m.attn)
                m.attn.forward = partial(attn_forward, m.attn)
                attns.update({n: m.attn})
            elif isinstance(m.attn, WindowedAttention):
                pass
            else:
                assert False
    else:
        _build = import_abspy("models", f"{os.path.dirname(__file__)}/../classification")
        build_mmpretrain_models = _build.build_mmpretrain_models
        model = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=False, with_norm=True,)
        model = model().cuda()
        from mmpretrain.models import VisionTransformer
        from mmpretrain.models.utils.attention import MultiheadAttention
        from mmpretrain.models.utils.attention import scaled_dot_product_attention_pyimpl

        def mattn_forward(self: MultiheadAttention, x):
            B, N, _ = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                    self.head_dims).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn_drop = self.attn_drop if self.training else 0.

            scale = q.size(-1)**0.5
            attn_weight = q @ k.transpose(-2, -1) / scale
            setattr(self, "__data__", attn_weight)

            x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
            x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

            x = self.proj(x)
            x = self.out_drop(self.gamma1(self.proj_drop(x)))

            if self.v_shortcut:
                x = v.squeeze(1) + x
            return x
        
        for n, l in model.backbone.layers.named_children():
            l.attn.forward = partial(mattn_forward, l.attn)
            attns.update({n: l.attn})
        
        deit_small_baseline = model

    print(attns)
    showpath = os.path.join(this_path, "show/deit")

    featHW = 32
    idxs_posxs_posys = [
        [0, 0.7, 0.3], [0, 0.2, 0.8], 
        [149, 0.7, 0.5], [149, 0.2, 0.4], 
        [162, 0.7, 0.4], [162, 0.4, 0.4], 
        [204, 0.3, 0.6], [204, 0.7, 0.2], 
        [273, 0.2, 0.6], [273, 0.9, 0.5],
        [309, 0.1, 0.7], [309, 0.9, 0.8],

    ]
    for idx, posx, posy in idxs_posxs_posys:
        img, label = dataset[idx]

        with torch.no_grad():
            deit_small_baseline(img[None].cuda())

        os.makedirs(f"{showpath}/{idx}_{posx}_{posy}", exist_ok=True)
        denormalize(img).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()

        for m0 in ["attn"]:
            for m1 in ["attn"]:
                aaa = getattr(attns['8'], "__data__")[0]
                aaa = ((aaa - aaa.min()) / (aaa.max() - aaa.min())).mean(dim=0)
                if aaa.shape[0] == featHW * featHW + 1:
                    aaa = aaa[1:, 1:]
                mask = aaa[int(posy * featHW) * int(featHW) + int(posx * featHW)].view(featHW, featHW)
                visualize_attnmap(mask, f"{showpath}/{idx}_{posx}_{posy}/{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)
    breakpoint()


if __name__ == "__main__":
    # main_deit()
    main_vssm()

