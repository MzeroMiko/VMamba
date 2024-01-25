from mmdet.models.backbones.swin import BaseModule, MODELS
from mmseg.models.backbones.swin import MODELS as MODELS_mmseg
from vmamba.vmamba import VSSM, VSSLayer
from torch import nn
import os
import torch
from torch.utils import checkpoint
from functools import partial

@MODELS.register_module()
class MMDET_VSSM(BaseModule, VSSM):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, ape=False, 
                 out_indices=(0, 1, 2, 3), pretrained=None, 
                 **kwargs,
        ):
        BaseModule.__init__(self)
        VSSM.__init__(self, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, depths=depths, 
                 dims=dims, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                 norm_layer=norm_layer, patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint, ape=ape, **kwargs)
        
        # add norm ===========================
        self.out_indices = out_indices
        for i in out_indices:
            layer = nn.LayerNorm(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
        
        # modify layer ========================
        def layer_forward(self: VSSLayer, x):
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            
            y = None
            if self.downsample is not None:
                y = self.downsample(x)

            return x, y

        for l in self.layers:
            l.forward = partial(layer_forward, l)

        # delete head ===-======================
        del self.head
        del self.avgpool
        del self.norm

        # load pretrained ======================
        if pretrained is not None:
            assert os.path.exists(pretrained)
            self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=""):
        _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
        print(f"Successfully load ckpt {ckpt}")
        incompatibleKeys = self.load_state_dict(_ckpt['model'], strict=False)
        print(incompatibleKeys)

    def forward(self, x):
        x = self.patch_embed(x)
        # if self.ape:
        #     x = x + self.absolute_pos_embed
        # x = self.pos_drop(x)

        outs = []
        y = x
        for i, layer in enumerate(self.layers):
            x, y = layer(y) # (B, H, W, C)
            if i in self.out_indices:
                norm_layer: nn.LayerNorm = getattr(self, f'outnorm{i}')
                out = norm_layer(x)
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


@MODELS_mmseg.register_module()
class MMSEG_VSSM(MMDET_VSSM):
    ...



