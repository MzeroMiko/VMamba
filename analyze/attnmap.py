# this is only a script !
import os
import torch
import torch.nn as nn
from functools import partial
from PIL import Image
from utils import visualize, get_dataset, AttnMamba, import_abspy, show_mask_on_image
HOME = os.environ["HOME"].rstrip("/")


def main_vssm(det_model=True, showpath= "show/vssmattnmap"):
    raw_attn = True
    stage = 2
    block_id = 1
    img_size = 512
    featHW = 32 # stage 2 so 32

    if not det_model:
        dataset = get_dataset(root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', img_size=img_size, crop=False)
        idxs_posxs_posys = [
            [72, 0.7, 0.3], [72, 0.2, 0.8], 
            [273, 0.7, 0.3], [282, 0.2, 0.8], 
            [282, 0.7, 0.3], [282, 0.2, 0.8], 
            [14602, 0.7, 0.3], [14602, 0.2, 0.8], 
            [17460, 0.6, 0.3], [17460, 0.2, 0.6], 
            [19256, 0.7, 0.3], [19256, 0.2, 0.3], 
            [47512, 0.7, 0.3], [47512, 0.3, 0.6], 
        ]
        # print([i for i, s in enumerate(dataset.samples) if "ILSVRC2012_val_00012107.JPEG" in s[0] ])
    else:
        # we want multiple objects, so we choose to use det model
        dataset = get_dataset(root='/media/Disk1/Dataset/MSCOCO2014/images/', img_size=img_size, ret="val2014", crop=False)
        idxs_posxs_posys = [
            [0, 0.3, 0.5], [0, 0.8, 0.8], 
            [149, 0.7, 0.5], [149, 0.2, 0.4], 
            [162, 0.7, 0.4], [162, 0.4, 0.4], 
            [204, 0.3, 0.6], [204, 0.7, 0.2], 
            [273, 0.2, 0.6], [273, 0.9, 0.5],
            [309, 0.1, 0.7], [309, 0.9, 0.8],
        ]
    
    # dataset = get_dataset(root='/media/Disk1/Dataset/ADEChallengeData2016/images/', img_size=img_size, ret="validation", crop=False)    
    
    vmamba = import_abspy("vmamba", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/models"))
    vssm: nn.Module = vmamba.vmamba_tiny_s1l8().cuda().eval()
    if det_model:
        vssm.load_state_dict(AttnMamba.convert_state_dict_from_mmdet(torch.load(open(f"{HOME}/Workspace/PylanceAware/ckpts/private/detection/vssm1/detection/mask_rcnn_vssm_fpn_coco_tiny_ms_3x_s/epoch_36.pth", "rb"), map_location="cpu")["state_dict"]), strict=False)
    else:
        vssm.load_state_dict(torch.load(open(f"{HOME}/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth", "rb"), map_location="cpu")["model"], strict=False)

    if raw_attn:
        setattr(vssm.layers[stage].blocks[block_id].op, "__DEBUG__", True)
        ss2ds = vssm.layers[stage].blocks[block_id].op
    else:
        [[ setattr(blk.op, "__DEBUG__", True)  for blk in layer.blocks] for layer in vssm.layers ]
        ss2ds = [[blk.op  for blk in layer.blocks] for layer in vssm.layers ]

    for idx, posx, posy in idxs_posxs_posys:
        img, label = dataset[idx]

        with torch.no_grad():
            out = vssm(img[None].cuda())
        print(out.argmax().item(), label, img.shape)
        os.makedirs(f"{showpath}/{idx}_{posx}_{posy}", exist_ok=True)
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()
        Image.fromarray((deimg * 255).to(torch.uint8).numpy()).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")
        visualize.draw_image_grid(
            Image.fromarray((deimg * 255).to(torch.uint8).numpy()),
            [(posx * img_size, posy * img_size, img_size / featHW, img_size / featHW,)]
        ).save(f"{showpath}/{idx}_{posx}_{posy}/imori_grid.jpg")
        # continue

        for m0 in ["a0", "a1", "a2", "a3", "all", "nall"]:
        # for m0 in ["ao0", "ao1", "ao2", "ao3", "a0", "a1", "a2", "a3", "all", "nall"]:
            for m1 in ["CB", "CwBw", "ww"]:
                aaa = AttnMamba.get_attnmap_mamba(ss2ds, stage, f"{m0}_norm_{m1}", raw_attn=True, block_id=block_id)
                # attention map
                # visualize.visualize_attnmap(aaa, f"{showpath}/{idx}_{posx}_{posy}/attn_{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)
                # diag attention map
                # visualize.visualize_attnmap(torch.diag(aaa).view(featHW, featHW), f"{showpath}/{idx}_{posx}_{posy}/attn_{m0}_norm_{m1}_diag.jpg", colorbar=False, sticks=False)
                # activation map
                mask = aaa[int(posy * featHW) * int(featHW) + int(posx * featHW)].view(featHW, featHW)
                visualize.visualize_attnmap(mask, f"{showpath}/{idx}_{posx}_{posy}/activation_{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)


def main_deit(det_model=True, showpath="show/deitdet"):
    raw_attn = True
    stage = 2
    block_id = 1
    img_size = 512
    featHW = 32 # stage 2 so 32

    if not det_model:
        dataset = get_dataset(root='/media/Disk1/Dataset/ImageNet_ILSVRC2012/val', img_size=img_size, crop=False)
        idxs_posxs_posys = [
            [0, 0.7, 0.3], [0, 0.2, 0.8], 
            [149, 0.7, 0.5], [149, 0.2, 0.4], 
            [162, 0.7, 0.4], [162, 0.4, 0.4], 
            [204, 0.3, 0.6], [204, 0.7, 0.2], 
            [273, 0.2, 0.6], [273, 0.9, 0.5],
            [309, 0.1, 0.7], [309, 0.9, 0.8],
        ]
    else:
        # we want multiple objects, so we choose to use det model
        dataset = get_dataset(root='/media/Disk1/Dataset/MSCOCO2014/images/', img_size=img_size, ret="val2014", crop=False)
        idxs_posxs_posys = [
            [0, 0.7, 0.3], [0, 0.2, 0.8], 
            [149, 0.7, 0.5], [149, 0.2, 0.4], 
            [162, 0.7, 0.4], [162, 0.4, 0.4], 
            [204, 0.3, 0.6], [204, 0.7, 0.2], 
            [273, 0.2, 0.6], [273, 0.9, 0.5],
            [309, 0.1, 0.7], [309, 0.9, 0.8],
        ]
    
    # dataset = get_dataset(root='/media/Disk1/Dataset/ADEChallengeData2016/images/', img_size=img_size, ret="validation", crop=False)    
    
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
                m.attn.forward = partial(attn_forward, m.attn)
                attns.update({n: m.attn})            
            else:
                assert False
    else:
        from utils import BuildModels
        model = BuildModels.build_deit_mmpretrain(with_ckpt=True, scale="small").cuda().eval()
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

    print(attns.keys())

    for idx, posx, posy in idxs_posxs_posys:
        img, label = dataset[idx]

        with torch.no_grad():
            deit_small_baseline(img[None].cuda())

        os.makedirs(f"{showpath}/{idx}_{posx}_{posy}", exist_ok=True)
        deimg = img.cpu() * torch.tensor([0.25, 0.25, 0.25]).view(-1, 1, 1) + torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
        deimg = deimg.permute(1, 2, 0).cpu()
        visualize.draw_image_grid(
            Image.fromarray((deimg * 255).to(torch.uint8).numpy()),
            [(posx * img_size, posy * img_size, img_size / featHW, img_size / featHW,)]
        ).save(f"{showpath}/{idx}_{posx}_{posy}/imori.jpg")

        for m0 in ["attn"]:
            for m1 in ["attn"]:
                aaa = getattr(attns['8'], "__data__")[0]
                aaa = ((aaa - aaa.min()) / (aaa.max() - aaa.min())).mean(dim=0)
                
                # attention map
                visualize.visualize_attnmap(aaa, f"{showpath}/{idx}_{posx}_{posy}/attn_{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)
                
                # activation map
                if aaa.shape[0] == featHW * featHW + 1:
                    aaa = aaa[1:, 1:]
                mask = aaa[int(posy * featHW) * int(featHW) + int(posx * featHW)].view(featHW, featHW)
                visualize.visualize_attnmap(mask, f"{showpath}/{idx}_{posx}_{posy}/activation_{m0}_norm_{m1}.jpg", colorbar=False, sticks=False)


if __name__ == "__main__":
    this_path = os.path.dirname(os.path.abspath(__file__))
    # main_deit(det_model=True, showpath="show/deitdet")
    # main_deit(det_model=False, showpath="show/deitcls")
    main_vssm(det_model=False, showpath="show/vssmcls")
    # main_vssm(det_model=True, showpath="show/vssmdet")




