import os
from functools import partial
import torch
import torch.nn as nn
from PIL import Image
import torch
import torch.nn as nn

from utils import visualize, get_dataset, AttnMamba, import_abspy, show_mask_on_image
visualize_attnmap = visualize.visualize_attnmap
visualize_attnmaps = visualize.visualize_attnmaps
attnmap_mamba = AttnMamba.attnmap_mamba

HOME = os.environ["HOME"].rstrip("/")
this_path = os.path.dirname(os.path.abspath(__file__))


def main_attnmap():
    dataset = get_dataset(root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', img_size=512, crop=False)
    dataset = get_dataset(root='/media/Disk1/Dataset/MSCOCO2014/images/', img_size=512, ret="val2014", crop=False)
    # dataset = get_dataset(root='/media/Disk1/Dataset/ADEChallengeData2016/images/', img_size=448, ret="validation", crop=False)    
    
    vmamba = import_abspy("vmamba", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/models"))
    VSSM: nn.Module = vmamba.VSSM    
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
    # vssm.load_state_dict(torch.load(open(f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth", "rb"), map_location="cpu")["model"], strict=False)
    vssm.load_state_dict(AttnMamba.convert_state_dict_from_mmdet(torch.load(open(f"{HOME}/Workspace/PylanceAware/ckpts/private/detection/vssm1/detection/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s/epoch_36.pth", "rb"), map_location="cpu")["state_dict"]), strict=False)
    
    ss2ds = []
    for layer in vssm.layers:
        _ss2ds = []
        for blk in layer.blocks:
            ss2d = blk.op
            setattr(ss2d, "__DEBUG__", True)
            _ss2ds.append(ss2d)
        ss2ds.append(_ss2ds)
    
    showpath = os.path.join(this_path, "show/vssmattnmap")

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
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()
        Image.fromarray((deimg * 255).to(torch.uint8).numpy()).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        # visualize.draw_image_grid(
        #     Image.fromarray((deimg * 255).to(torch.uint8).numpy()),
        #     [(posx * 512, posy * 512, 512 / 32, 512 / 32)]
        # ).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        # continue

        for m0 in ["ao0", "ao1", "ao2", "ao3", "a0", "a1", "a2", "a3", "all", "nall"]:
            for m1 in ["CB", "CwBw", "ww"]:
                aaa = AttnMamba.get_attnmap_mamba(ss2ds, 2, f"{m0}_norm_{m1}", raw_attn=True, block_id=1)
                # mask = aaa[int(posy * featHW) * int(featHW) + int(posx * featHW)].view(featHW, featHW)
                visualize_attnmap(aaa, f"{showpath}/{idx}_{posx}_{posy}/{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)
                visualize_attnmap(torch.diag(aaa).view(featHW, featHW), f"{showpath}/{idx}_{posx}_{posy}/{m0}_norm_{m1}_diag.jpg", colorbar=False, sticks=False)
        # breakpoint()
    breakpoint()



def main_vssm():
    dataset = get_dataset(root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', img_size=512, crop=False)
    dataset = get_dataset(root='/media/Disk1/Dataset/MSCOCO2014/images/', img_size=512, ret="val2014", crop=False)
    # dataset = get_dataset(root='/media/Disk1/Dataset/ADEChallengeData2016/images/', img_size=448, ret="validation", crop=False)    
    
    vmamba = import_abspy("vmamba", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/models"))
    VSSM: nn.Module = vmamba.VSSM    
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
    vssm.load_state_dict(torch.load(open(f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth", "rb"), map_location="cpu")["model"], strict=False)
    # vssm.load_state_dict(AttnMamba.convert_state_dict_from_mmdet(torch.load(open(f"{HOME}/Workspace/PylanceAware/ckpts/private/detection/vssm1/detection/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s/epoch_36.pth", "rb"), map_location="cpu")["state_dict"]), strict=False)
    
    ss2ds = []
    for layer in vssm.layers:
        _ss2ds = []
        for blk in layer.blocks:
            ss2d = blk.op
            setattr(ss2d, "__DEBUG__", True)
            _ss2ds.append(ss2d)
        ss2ds.append(_ss2ds)
    
    showpath = os.path.join(this_path, "show/vssm_cls")

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
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()
        visualize.draw_image_grid(
            Image.fromarray((deimg * 255).to(torch.uint8).numpy()),
            [(posx * 512, posy * 512, 512 / 32, 512 / 32)]
        ).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        # continue

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
        _deit = import_abspy("vit_adpter_baseline", f"{HOME}/Workspace/PylanceAware/ckpts/ckpts")
        # from ckpts.ckpts.vit_adpter_baseline import deit_small_baseline, Attention, WindowedAttention
        deit_small_baseline, Attention, WindowedAttention = _deit.deit_small_baseline, _deit.Attention, _deit.WindowedAttention
        sd = torch.load(f"{HOME}/Workspace/PylanceAware/ckpts/others/deit_small_patch16_224-cd65a155.pth", map_location=torch.device("cpu"))
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
            if isinstance(m.attn, WindowedAttention):
                pass
            elif isinstance(m.attn, Attention):
                print(n, m.attn)
                m.attn.forward = partial(attn_forward, m.attn)
                attns.update({n: m.attn})            
            else:
                assert False
        showpath = os.path.join(this_path, "show/deitdet")
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
        showpath = os.path.join(this_path, "show/deitcls")

    print(attns)

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
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()
        visualize.draw_image_grid(
            Image.fromarray((deimg * 255).to(torch.uint8).numpy()),
            [(posx * 512, posy * 512, 512 / 32, 512 / 32)]
        ).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        # continue

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
    # main_attnmap()
    main_deit(det_model=True)
    main_vssm()

