# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import VisionTransformer

"""Vision Transformer (ViT) in PyTorch.

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, Mlp, to_2tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, H, W


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0., window_size=14):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x, H, W):
        B, N, C = x.shape
        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size

        qkv = self.qkv(x)  # [B, N, C]
        qkv = qkv.transpose(1, 2).reshape(B, C * 3, H, W)  # [B, C, H, W]
        qkv = F.pad(qkv, [0, W_ - W, 0, H_ - H], mode='constant')

        qkv = F.unfold(qkv, kernel_size=(self.window_size, self.window_size),
                       stride=(self.window_size, self.window_size))
        B, C_kw_kw, L = qkv.shape  # L - the num of windows
        qkv = qkv.reshape(B, C * 3, N_, L).permute(0, 3, 2, 1)  # [B, L, N_, C]
        qkv = qkv.reshape(B, L, N_, 3, self.num_heads, C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # q,k,v [B, L, num_head, N_, C/num_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, L, num_head, N_, N_]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        # attn @ v = [B, L, num_head, N_, C/num_head]
        x = (attn @ v).permute(0, 2, 4, 3, 1).reshape(B, C_kw_kw // 3, L)

        x = F.fold(x, output_size=(H_, W_),
                   kernel_size=(self.window_size, self.window_size),
                   stride=(self.window_size, self.window_size))  # [B, C, H_, W_]
        x = x[:, :, :H, :W].reshape(B, C, N).transpose(-1, -2)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# class WindowedAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=14, pad_mode="constant"):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.window_size = window_size
#         self.pad_mode = pad_mode
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#
#         N_ = self.window_size * self.window_size
#         H_ = math.ceil(H / self.window_size) * self.window_size
#         W_ = math.ceil(W / self.window_size) * self.window_size
#         x = x.view(B, H, W, C)
#         x = F.pad(x, [0, 0, 0, W_ - W, 0, H_- H], mode=self.pad_mode)
#
#         x = window_partition(x, window_size=self.window_size)# nW*B, window_size, window_size, C
#         x = x.view(-1, N_, C)
#
#         qkv = self.qkv(x).view(-1, N_, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
#         attn = (q @ k.transpose(-2, -1)) * self.scale # [B, L, num_head, N_, N_]
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn) # [B, L, num_head, N_, N_]
#         x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)
#
#         x = window_reverse(x, self.window_size, H_, W_)
#         x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm=LayerNorm,
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = norm(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(bottleneck_channels,
                               bottleneck_channels,
                               3,
                               padding=1,
                               bias=False,)
        self.norm2 = norm(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = norm(out_channels)

        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in [self.conv1, self.norm1, self.act1,
                      self.conv2, self.norm2, self.act2,
                      self.conv3, self.norm3]:
            x = layer(x)

        out = x + out
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., with_cp=False,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 windowed=False, window_size=14, use_residual=False, layer_scale=False):
        super().__init__()
        self.with_cp = with_cp
        self.use_residual = use_residual
        self.norm1 = norm_layer(dim)
        if windowed:
            self.attn = WindowedAttention(dim, num_heads=num_heads,
                                          qkv_bias=qkv_bias, attn_drop=attn_drop,
                                          proj_drop=drop, window_size=window_size)
        else:
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                  attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)
            
        if self.use_residual:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm=LayerNorm,
                act_layer=act_layer,
            )
            
    def forward(self, x, H, W):
        
        def _inner_forward(x):
            if self.layer_scale:
                x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                
            if self.use_residual:
                B, N, C = x.shape
                x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = self.residual(x)
                x = x.permute(0, 2, 3, 1).reshape(B, N, C)
                
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        
        return x


class TIMMVisionTransformer(nn.Module):
    """Vision Transformer.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, residual_indices=[], embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., layer_scale=True, embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, window_attn=False, window_size=14, with_cp=False, pretrained=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            pretrained: (str): pretrained path
            with_cp: (bool): use checkpoint or not
        """
        super().__init__()
        # self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer
        self.act_layer = act_layer
        self.pretrain_size = img_size
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate

        window_attn = [window_attn] * depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * depth if not isinstance(window_size, list) else window_size
        logging.info('window attention:', window_attn)
        logging.info('window size:', window_size)
        logging.info('layer scale:', layer_scale)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                  windowed=window_attn[i], window_size=window_size[i],
                  layer_scale=layer_scale, with_cp=with_cp,
                  use_residual=True if i in residual_indices else False) for i in range(depth)
        ])

        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x


class ViTBaseline(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, out_indices=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.out_indices = out_indices
        assert out_indices is not None

        embed_dim = self.embed_dim
        self.norm1 = self.norm_layer(embed_dim)
        self.norm2 = self.norm_layer(embed_dim)
        self.norm3 = self.norm_layer(embed_dim)
        self.norm4 = self.norm_layer(embed_dim)

        self.up1 = nn.Sequential(*[
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        ])
        self.up2 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.up3 = nn.Identity()
        self.up4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up1.apply(self._init_weights)
        self.up2.apply(self._init_weights)
        self.up3.apply(self._init_weights)
        self.up4.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, ResBottleneckBlock):
            m.norm3.weight.data.zero_()
            m.norm3.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward_features(self, x):
        outs = []
        x, H, W = self.patch_embed(x)
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        for index, blk in enumerate(self.blocks):
            x = blk(x, H, W)
            if index in self.out_indices:
                outs.append(x)
        return outs, H, W

    def forward(self, x):
        outs, H, W = self.forward_features(x)
        if len(outs) == 1: # for ViTDet
            f1 = f2 = f3 = f4 = outs[0]
        else: # for ViT
            f1, f2, f3, f4 = outs
        bs, n, dim = f1.shape

        # Final Norm
        f1 = self.norm1(f1).transpose(1, 2).reshape(bs, dim, H, W)
        f2 = self.norm2(f2).transpose(1, 2).reshape(bs, dim, H, W)
        f3 = self.norm3(f3).transpose(1, 2).reshape(bs, dim, H, W)
        f4 = self.norm4(f4).transpose(1, 2).reshape(bs, dim, H, W)

        f1 = self.up1(f1).contiguous()
        f2 = self.up2(f2).contiguous()
        f3 = self.up3(f3).contiguous()
        f4 = self.up4(f4).contiguous()

        return [f1, f2, f3, f4]


# https://github.com/czczup/ViT-Adapter/blob/main/detection/configs/mask_rcnn/mask_rcnn_deit_small_fpn_3x_coco.py
deit_small_baseline = partial(ViTBaseline,
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    drop_path_rate=0.1,
    out_indices=[2, 5, 8, 11],
    window_attn=[True, True, False, True, True, False,
                 True, True, False, True, True, False],
    window_size=[14, 14, None, 14, 14, None,
                 14, 14, None, 14, 14, None],
)


