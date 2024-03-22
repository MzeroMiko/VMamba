# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the MIT license
"""ConvNext TIMM version with S4ND integration.

Paper: `A ConvNet for the 2020s` - https://arxiv.org/pdf/2201.03545.pdf
Original code and weights from https://github.com/facebookresearch/ConvNeXt, original copyright below
Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""

from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_module
# from timm.models.helpers import named_apply, build_model_with_cfg, checkpoint_seq
from timm.models.helpers import named_apply, build_model_with_cfg
from timm.models.layers import trunc_normal_, ClassifierHead, SelectAdaptivePool2d, DropPath, ConvMlp, Mlp
from timm.models.registry import register_model

import copy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from omegaconf import OmegaConf

# S4 imports
import src.utils as utils
import src.utils.registry as registry
from src.models.nn import TransposedLinear

__all__ = ['ConvNeXt']  # model_registry will add each entrypoint fn to this


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.0', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = dict(
    convnext_tiny=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth"),
    convnext_small=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth"),
    convnext_base=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth"),
    convnext_large=_cfg(url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth"),

    convnext_nano_hnf=_cfg(url=''),
    convnext_tiny_hnf=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_tiny_hnf_a2h-ab7e9df2.pth',
        crop_pct=0.95),

    convnext_tiny_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth'),
    convnext_small_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth'),
    convnext_base_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth'),
    convnext_large_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth'),
    convnext_xlarge_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth'),

    convnext_tiny_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_small_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_base_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_large_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),
    convnext_xlarge_384_in22ft1k=_cfg(
        url='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth',
        input_size=(3, 384, 384), pool_size=(12, 12), crop_pct=1.0),

    convnext_tiny_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth", num_classes=21841),
    convnext_small_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth", num_classes=21841),
    convnext_base_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth", num_classes=21841),
    convnext_large_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth", num_classes=21841),
    convnext_xlarge_in22k=_cfg(
        url="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth", num_classes=21841),
)


def _is_contiguous(tensor: torch.Tensor) -> bool:
    # jit is oh so lovely :/
    # if torch.jit.is_tracing():
    #     return True
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


def get_num_layer_for_convnext(var_name, variant='tiny'):
    """
    Divide [3, 3, 27, 3] layers into 12 groups; each group is three
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    num_max_layer = 12
    if "stem" in var_name:
        return 0

    # note:  moved norm_layer outside of downsample module
    elif "downsample" in var_name or "norm_layer" in var_name:
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif "stages" in var_name:
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[4])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            if variant == 'tiny':
                layer_id = 3 + block_id
            else:
                layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    else:
        return num_max_layer + 1


def get_num_layer_for_convnext_tiny(var_name):
    return get_num_layer_for_convnext(var_name, 'tiny')


@register_notrace_module
class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True):
        """ tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        For some reason tie=False is dog slow, prob something wrong with torch.distribution
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            return X * mask * (1.0/(1-self.p))
        return X

@register_notrace_module
class LayerNorm2d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x

@register_notrace_module
class LayerNorm3d(nn.LayerNorm):
    r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, L, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 4, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 4, 1, 2, 3)
        else:
            s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None, None] + self.bias[:, None, None, None]
            return x

@register_notrace_module
class TransposedLN(nn.Module):
    def __init__(self, d, scalar=True):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(1))
        self.s = nn.Parameter(torch.ones(1))
        setattr(self.m, "_optim", {"weight_decay": 0.0})
        setattr(self.s, "_optim", {"weight_decay": 0.0})

    def forward(self, x):
        s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
        y = (self.s/s) * (x-m+self.m)
        return y


class Conv2dWrapper(nn.Module):
    """
    Light wrapper used to just absorb the resolution flag (like s4's conv layer)
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, **kwargs)

    def forward(self, x, resolution=None):
        return self.conv(x)


class S4DownSample(nn.Module):
    """ S4 conv block with downsampling using avg pool

    Args:
        downsample_layer (dict): config for creating s4 layer
        in_ch (int): num input channels
        out_ch (int): num output channels
        stride (int): downsample factor in avg pool

    """
    def __init__(self, downsample_layer, in_ch, out_ch, stride=1, activate=False, glu=False, pool3d=False):
        super().__init__()

        # create s4
        self.s4conv = utils.instantiate(registry.layer, downsample_layer, in_ch)
        self.act = nn.GELU() if activate else nn.Identity()
        if pool3d:
            self.avgpool = nn.AvgPool3d(kernel_size=stride, stride=stride)
        else:
            self.avgpool = nn.AvgPool2d(kernel_size=stride, stride=stride)
        self.glu = glu
        d_out = 2*out_ch if self.glu else out_ch
        self.fc = TransposedLinear(in_ch, d_out)

    def forward(self, x, resolution=1):
        x = self.s4conv(x, resolution)
        x = self.act(x)
        x = self.avgpool(x)
        x = self.fc(x)
        if self.glu:
            x = F.glu(x, dim=1)
        return x


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block

    # previous convnext notes:
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    # two options for convs are:
        - conv2d, depthwise (original)
        - s4nd, used if a layer config passed

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        layer (config/dict): config for s4 layer
    """

    def __init__(self,
            dim,
            drop_path=0.,
            ls_init_value=1e-6,
            conv_mlp=False,
            mlp_ratio=4,
            norm_layer=None,
            layer=None,
        ):
        super().__init__()

        assert norm_layer is not None
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp

        # Depthwise conv
        if layer is None:
            self.conv_dw = Conv2dWrapper(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        else:
            self.conv_dw = utils.instantiate(registry.layer, layer, dim)

        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, resolution=1):
        shortcut = x

        x = self.conv_dw(x, resolution)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):

    def __init__(self,
        stem_type='patch',  # regular convnext
        in_ch=3,
        out_ch=64,
        img_size=None,
        patch_size=4,
        stride=4,
        stem_channels=32,
        stem_layer=None,
        stem_l_max=None,
        downsample_act=False,
        downsample_glu=False,
        norm_layer=None,
    ):
        super().__init__()

        self.stem_type = stem_type

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        self.pre_stem = None
        self.post_stem = None
        if stem_type == 'patch':
            print("stem type: ", 'patch')
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size),
                norm_layer(out_ch)
            )

        elif stem_type == 'depthwise_patch':
            print("stem type: ", 'depthwise_patch')
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, stem_channels, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(stem_channels, stem_channels, kernel_size=patch_size, stride=1, padding='same', groups=stem_channels),
                nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
                TransposedLinear(stem_channels, 2*out_ch),
                nn.GLU(dim=1),
                norm_layer(out_ch),
            )

        elif stem_type == 'new_patch':
            print("stem type: ", 'new_patch')
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, stem_channels, kernel_size=patch_size, stride=1, padding='same'),
                nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
                TransposedLinear(stem_channels, 2*out_ch),
                nn.GLU(dim=1),
                norm_layer(out_ch),
            )

        elif stem_type == 'new_s4nd_patch':
            print("stem type: ", 'new_s4nd_patch')
            stem_layer_copy = copy.deepcopy(stem_layer)
            assert stem_l_max is not None, "need to provide a stem_l_max to use stem=new_s4nd_patch"
            stem_layer_copy["l_max"] = stem_l_max

            self.pre_stem = nn.Identity()
            self.stem = utils.instantiate(registry.layer, stem_layer_copy, in_ch, out_channels=stem_channels)
            self.post_stem = nn.Sequential(
                nn.AvgPool2d(kernel_size=patch_size, stride=patch_size),
                TransposedLinear(stem_channels, 2*out_ch),
                nn.GLU(dim=1),
                norm_layer(out_ch)
            )

        elif stem_type == 's4nd_patch':
            print("stem type: ", "s4nd_patch")
            stem_layer_copy = copy.deepcopy(stem_layer)
            stem_layer_copy["l_max"] = img_size

            self.pre_stem = nn.Conv2d(in_ch, stem_channels, kernel_size=1, stride=1, padding=0)
            # s4 + norm + avg pool + linear
            self.stem = S4DownSample(stem_layer_copy, stem_channels, out_ch, stride=patch_size, activate=downsample_act, glu=downsample_glu)
            self.post_stem = norm_layer(out_ch)

        elif stem_type == 's4nd':
            # mix of conv2d + s4
            print("stem type: ", 's4nd')
            stem_layer_copy = copy.deepcopy(stem_layer)
            stem_layer_copy["l_max"] = img_size

            # s4_downsample = nn.Sequential(
            #     utils.instantiate(registry.layer, stage_layer_copy, stem_channels),
            #     nn.AvgPool2d(kernel_size=2, stride=2),
            #     TransposedLinear(stem_channels, 64),
            # )
            s4_downsample = S4DownSample(stem_layer_copy, stem_channels, 64, stride=2, activate=downsample_act, glu=downsample_glu)
            self.pre_stem = nn.Sequential(
                nn.Conv2d(in_ch, stem_channels, kernel_size=1, stride=1, padding=0),
                norm_layer(stem_channels),
                nn.GELU()
            )
            self.stem = s4_downsample
            self.post_stem = nn.Identity()

        # regular strided conv downsample
        elif stem_type == 'default':
            print("stem type: DEFAULT. Make sure this is what you want.")
            self.stem = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
                norm_layer(32),
                nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
            )

        else:
            raise NotImplementedError("provide a valid stem type!")

    def forward(self, x, resolution):
        # if using s4nd layer, need to pass resolution
        if self.stem_type in ['s4nd', 's4nd_patch', 'new_s4nd_patch']:
            x = self.pre_stem(x)
            x = self.stem(x, resolution)
            x = self.post_stem(x)
        else:
            x = self.stem(x)
        return x


class ConvNeXtStage(nn.Module):

    """
    Will create a stage, made up of downsampling and conv blocks.
    There are 2 choices for each of these:
        downsampling: s4 or strided conv (original)
        conv stage:  s4 or conv2d (original)
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            img_size=None,
            stride=2,
            depth=2,
            dp_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            norm_layer=None,
            cl_norm_layer=None,
            # cross_stage=False,
            stage_layer=None, # config
            # downsample_layer=None,
            downsample_type=None,
            downsample_act=False,
            downsample_glu=False,
        ):
        super().__init__()

        self.grad_checkpointing = False
        self.downsampling = False

        # 2 options to downsample
        if in_chs != out_chs or stride > 1:
            self.downsampling = True
            # s4 type copies config from corresponding stage layer
            if downsample_type == 's4nd':
                print("s4nd downsample")
                downsample_layer_copy = copy.deepcopy(stage_layer)
                downsample_layer_copy["l_max"] = img_size  # always need to update curr l_max
                self.norm_layer = norm_layer(in_chs)

                # mimics strided conv but w/s4
                self.downsample = S4DownSample(downsample_layer_copy, in_chs, out_chs, stride=stride, activate=downsample_act, glu=downsample_glu)

            # strided conv
            else:
                print("strided conv downsample")
                self.norm_layer = norm_layer(in_chs)
                self.downsample = Conv2dWrapper(in_chs, out_chs, kernel_size=stride, stride=stride)
        # else:
        #     self.norm_layer = nn.Identity()
        #     self.downsample = nn.Identity()

        if stage_layer is not None:
            stage_layer["l_max"] = [x // stride for x in img_size]

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.ModuleList()

        for j in range(depth):
            self.blocks.append(
            ConvNeXtBlock(
                dim=out_chs, drop_path=dp_rates[j], ls_init_value=ls_init_value, conv_mlp=conv_mlp,
                norm_layer=norm_layer if conv_mlp else cl_norm_layer, layer=stage_layer)
            )

    def forward(self, x, resolution=1):

        if self.downsampling:
            x = self.norm_layer(x)
            x = self.downsample(x, resolution)
            for block in self.blocks:
                x = block(x, resolution)
        # not downsampling we just don't create a downsample layer, since before Identity can't accept pass through args
        else:
            for block in self.blocks:
                x = block(x, resolution)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_head (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            output_stride=32,
            patch_size=4,
            stem_channels=8,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            ls_init_value=1e-6,
            conv_mlp=False, # whether to transpose channels to last dim inside MLP
            stem_type='patch',  # supports `s4nd` + avg pool
            stem_l_max=None,  # len of l_max in stem (if using s4)
            downsample_type='patch',  # supports `s4nd` + avg pool
            downsample_act=False,
            downsample_glu=False,
            head_init_scale=1.,
            head_norm_first=False,
            norm_layer=None,
            custom_ln=False,
            drop_head=0.,
            drop_path_rate=0.,
            layer=None, # Shared config dictionary for the core layer
            stem_layer=None,
            stage_layers=None,
            img_size=None,
            # **kwargs,  # catch all
    ):
        super().__init__()

        assert output_stride == 32
        if norm_layer is None:
            if custom_ln:
                norm_layer = TransposedLN
            else:
                norm_layer = partial(LayerNorm2d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_head = drop_head
        self.feature_info = []
        self._img_sizes = [img_size]

        # Broadcast dictionaries
        if layer is not None:
            stage_layers = [OmegaConf.merge(layer, s) for s in stage_layers]
            stem_layer = OmegaConf.merge(layer, stem_layer)

        # instantiate stem
        self.stem = Stem(
            stem_type=stem_type,
            in_ch=in_chans,
            out_ch=dims[0],
            img_size=img_size,
            patch_size=patch_size,
            stride=patch_size,
            stem_channels=stem_channels,
            stem_layer=stem_layer,
            stem_l_max=stem_l_max,
            downsample_act=downsample_act,
            downsample_glu=downsample_glu,
            norm_layer=norm_layer,
        )

        if stem_type == 's4nd' or stem_type == 'default':
            stem_stride = 2
            prev_chs = 64
        else:
            stem_stride = patch_size
            prev_chs = dims[0]

        curr_img_size = [x // stem_stride for x in img_size]
        self._img_sizes.append(curr_img_size)

        self.stages = nn.ModuleList()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            # if stem downsampled by 4, then in stage 0, we don't downsample
            # if stem downsampled by 2, then in stage 0, we downsample by 2
            # all other stages we downsample by 2 no matter what
            stride = 1 if i==0 and stem_stride == 4 else 2  # stride 1 is no downsample (because already ds in stem)

            # print("stage {}, before downsampled img size {}, stride {}".format(i, curr_img_size, stride))

            out_chs = dims[i]
            self.stages.append(ConvNeXtStage(
                prev_chs,
                out_chs,
                img_size=curr_img_size,
                stride=stride,
                depth=depths[i],
                dp_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                norm_layer=norm_layer,
                cl_norm_layer=cl_norm_layer,
                stage_layer=stage_layers[i],
                downsample_type=downsample_type,
                downsample_act=downsample_act,
                downsample_glu=downsample_glu,
                )
            )

            prev_chs = out_chs
            curr_img_size = [x // stride for x in curr_img_size]  # update image size for next stage
            self._img_sizes.append(curr_img_size)

            # # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            # self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        # self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
                ('drop', nn.Dropout(self.drop_head)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        if global_pool is not None:
            self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, resolution=1):
        x = self.stem(x, resolution)
        for stage in self.stages:
            x = stage(x, resolution)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x, resolution=1, state=None):
        x = self.forward_features(x, resolution)
        x = self.forward_head(x)
        return x, None


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)

        # check if has bias first
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """ Remap FB checkpoints -> timm """
    if 'head.norm.weight' in state_dict or 'norm_pre.weight' in state_dict:
        return state_dict  # non-FB checkpoint
    if 'model' in state_dict:
        state_dict = state_dict['model']
    out_dict = {}
    import re
    for k, v in state_dict.items():
        k = k.replace('downsample_layers.0.', 'stem.')
        k = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', k)
        k = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', k)
        k = k.replace('dwconv', 'conv_dw')
        k = k.replace('pwconv', 'mlp.fc')
        k = k.replace('head.', 'head.fc.')
        if k.startswith('norm.'):
            k = k.replace('norm', 'head.norm')
        if v.ndim == 2 and 'head' not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v
    return out_dict


def _create_convnext(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        ConvNeXt, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs)
    return model


# @register_model
# def convnext_nano_hnf(pretrained=False, **kwargs):
#     model_args = dict(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640), head_norm_first=True, conv_mlp=True, **kwargs)
#     model = _create_convnext('convnext_nano_hnf', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_tiny_hnf(pretrained=False, **kwargs):
#     model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True, **kwargs)
#     model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_tiny_hnfd(pretrained=False, **kwargs):
#     model_args = dict(
#         depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), head_norm_first=True, conv_mlp=True, stem_type='dual', **kwargs)
#     model = _create_convnext('convnext_tiny_hnf', pretrained=pretrained, **model_args)
#     return model

@register_model
def convnext_micro(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 3, 3), dims=(64, 128, 256, 512), **kwargs)
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model

@register_model
def convnext_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext('convnext_tiny', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_small(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    model = _create_convnext('convnext_small', pretrained=pretrained, **model_args)
    return model


@register_model
def convnext_base(pretrained=False, **kwargs):
    model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    model = _create_convnext('convnext_base', pretrained=pretrained, **model_args)
    return model


# @register_model
# def convnext_large(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     model = _create_convnext('convnext_large', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_tiny_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_tiny_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_small_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_small_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_base_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_base_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_large_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     model = _create_convnext('convnext_large_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_xlarge_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     model = _create_convnext('convnext_xlarge_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_tiny_384_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_tiny_384_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_small_384_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_small_384_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_base_384_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_base_384_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_large_384_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     model = _create_convnext('convnext_large_384_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_xlarge_384_in22ft1k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     model = _create_convnext('convnext_xlarge_384_in22ft1k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_tiny_in22k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_tiny_in22k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_small_in22k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_small_in22k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_base_in22k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
#     model = _create_convnext('convnext_base_in22k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_large_in22k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
#     model = _create_convnext('convnext_large_in22k', pretrained=pretrained, **model_args)
#     return model


# @register_model
# def convnext_xlarge_in22k(pretrained=False, **kwargs):
#     model_args = dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
#     model = _create_convnext('convnext_xlarge_in22k', pretrained=pretrained, **model_args)
#     return model


class Conv3d(nn.Conv3d):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding=0, groups=1, factor=False):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, groups=groups)
        self.factor = factor
        self.in_ch=in_ch
        self.out_ch=out_ch
        self.kernel_size=[kernel_size] if isinstance(kernel_size, int) else kernel_size
        self.stride=stride
        self.padding=padding
        self.groups=groups

        if self.factor:
            self.weight = nn.Parameter(self.weight[:, :, 0, :, :]) # Subsample time dimension
            self.time_weight = nn.Parameter(self.weight.new_ones(self.kernel_size[0]) / self.kernel_size[0])
        else:
            pass

    def forward(self, x):
        if self.factor:
            weight = self.weight[:, :, None, :, :] * self.time_weight[:, None, None]
            y = F.conv3d(x, weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.groups)
        else:
            y = super().forward(x)
        return y


class Conv3dWrapper(nn.Module):
    """
    Light wrapper to make consistent with 2d version (allows for easier inflation).
    """
    def __init__(self, dim_in, dim_out, **kwargs):
        super().__init__()
        self.conv = Conv3d(dim_in, dim_out, **kwargs)

    def forward(self, x, resolution=None):
        return self.conv(x)


class ConvNeXtBlock3D(nn.Module):
    """ ConvNeXt Block

    # previous convnext notes:
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.

    # two options for convs are:
        - conv2d, depthwise (original)
        - s4nd, used if a layer config passed

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        layer (config/dict): config for s4 layer
    """

    def __init__(self,
            dim,
            drop_path=0.,
            drop_mlp=0.,
            ls_init_value=1e-6,
            conv_mlp=False,
            mlp_ratio=4,
            norm_layer=None,
            block_tempor_kernel=3,
            layer=None,
            factor_3d=False,
        ):
        super().__init__()

        assert norm_layer is not None
        # if not norm_layer:
        #     norm_layer = partial(LayerNorm2d, eps=1e-6) if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        mlp_layer = ConvMlp if conv_mlp else Mlp
        self.use_conv_mlp = conv_mlp

        # Depthwise conv
        if layer is None:
            tempor_padding = block_tempor_kernel // 2  # or 2

            # self.conv_dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
            self.conv_dw = Conv3dWrapper(
                dim,
                dim,
                kernel_size=(block_tempor_kernel, 7, 7),
                padding=(tempor_padding, 3, 3),
                stride=(1, 1, 1),
                groups=dim,
                factor=factor_3d,
            )  # depthwise conv
        else:
            self.conv_dw = utils.instantiate(registry.layer, layer, dim)

        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=nn.GELU, drop=drop_mlp)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 4, 1, 2, 3)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage3D(nn.Module):

    """
    Will create a stage, made up of downsampling and conv blocks.
    There are 2 choices for each of these:
        downsampling: s4 or strided conv (original)
        conv stage:  s4 or conv2d (original)
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            video_size=None,  # L, H, W
            stride=(2, 2, 2), # Strides for L, H, W
            depth=2,
            dp_rates=None,
            ls_init_value=1.0,
            conv_mlp=False,
            norm_layer=None,
            cl_norm_layer=None,
            stage_layer=None, # config
            block_tempor_kernel=3,
            downsample_type=None,
            downsample_act=False,
            downsample_glu=False,
            factor_3d=False,
            drop_mlp=0.,
        ):
        super().__init__()

        self.grad_checkpointing = False

        # 2 options to downsample
        if in_chs != out_chs or np.any(np.array(stride) > 1):
            # s4 type copies config from corresponding stage layer
            if downsample_type == 's4nd':
                print("s4nd downsample")
                downsample_layer_copy = copy.deepcopy(stage_layer)
                downsample_layer_copy["l_max"] = video_size  # always need to update curr l_max
                self.norm_layer = norm_layer(in_chs)
                # mimics strided conv but w/s4
                self.downsample = S4DownSample(
                                    downsample_layer_copy,
                                    in_chs,
                                    out_chs,
                                    stride=stride,
                                    activate=downsample_act,
                                    glu=downsample_glu,
                                    pool3d=True,
                                )
                # self.downsample = nn.Sequential(
                #     norm_layer(in_chs),
                #     S4DownSample(
                #         downsample_layer_copy,
                #         in_chs,
                #         out_chs,
                #         stride=stride,
                #         activate=downsample_act,
                #         glu=downsample_glu,
                #         pool3d=True,
                #     )
                # )
            # strided conv
            else:
                print("strided conv downsample")
                self.norm_layer = norm_layer(in_chs)
                self.downsample = Conv3dWrapper(in_chs, out_chs, kernel_size=stride, stride=stride, factor=factor_3d)

                # self.downsample = nn.Sequential(
                #     norm_layer(in_chs),
                #     Conv3d(in_chs, out_chs, kernel_size=stride, stride=stride, factor=factor_3d),
                # )

        else:
            self.norm_layer = nn.Identity()
            self.downsample = nn.Identity()

        if stage_layer is not None:
            stage_layer["l_max"] = [
                x // stride if isinstance(stride, int) else x // stride[i]
                for i, x in enumerate(video_size)
            ]

        dp_rates = dp_rates or [0.] * depth
        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3D(
                dim=out_chs,
                drop_path=dp_rates[j],
                drop_mlp=drop_mlp,
                ls_init_value=ls_init_value,
                conv_mlp=conv_mlp,
                norm_layer=norm_layer if conv_mlp else cl_norm_layer,
                block_tempor_kernel=block_tempor_kernel,
                layer=stage_layer,
                factor_3d=factor_3d,
            )
            for j in range(depth)
        ])

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class Stem3d(nn.Module):

    def __init__(self,
            stem_type='patch',  # supports `s4nd` + avg pool
            in_chans=3,
            spatial_patch_size=4,
            tempor_patch_size=4,
            stem_channels=8,
            dims=(96, 192, 384, 768),  
            stem_l_max=None,  # len of l_max in stem (if using s4)
            norm_layer=None,
            custom_ln=False,
            layer=None, # Shared config dictionary for the core layer
            stem_layer=None,
            factor_3d=False,
    ):
        super().__init__()

        self.stem_type = stem_type

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        if stem_type == 'patch':
            print("stem type: ", 'patch')
            kernel_3d = [tempor_patch_size, spatial_patch_size, spatial_patch_size]
            self.stem = nn.Sequential(
                Conv3d(
                    in_chans,
                    dims[0],
                    kernel_size=kernel_3d,
                    stride=kernel_3d,
                    factor=factor_3d,
                ),
                norm_layer(dims[0]),
            )
        elif stem_type == 'new_s4nd_patch':
            print("stem type: ", 'new_s4nd_patch')
            stem_layer_copy = copy.deepcopy(stem_layer)
            assert stem_l_max is not None, "need to provide a stem_l_max to use stem=new_s4nd_patch"
            stem_layer_copy["l_max"] = stem_l_max
            s4_ds = utils.instantiate(registry.layer, stem_layer_copy, in_chans, out_channels=stem_channels)
            kernel_3d = [tempor_patch_size, spatial_patch_size, spatial_patch_size]
            self.stem = nn.Sequential(
                s4_ds,
                nn.AvgPool3d(kernel_size=kernel_3d, stride=kernel_3d),
                TransposedLinear(stem_channels, 2*dims[0]),
                nn.GLU(dim=1),
                norm_layer(dims[0]),
            )
        else:
            raise NotImplementedError("provide a valid stem type!")

    def forward(self, x, resolution=None):
        # if using s4nd layer, need to pass resolution
        if self.stem_type in ['new_s4nd_patch']:
            x = self.stem(x, resolution)
        else:
            x = self.stem(x)
        return x


class ConvNeXt3D(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_head (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            # global_pool='avg',
            spatial_patch_size=4,
            tempor_patch_size=4,
            output_spatial_stride=32,
            # patch_size=(1, 4, 4),
            stem_channels=8,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            ls_init_value=1e-6,
            conv_mlp=False, # whether to transpose channels to last dim inside MLP
            stem_type='patch',  # supports `s4nd` + avg pool
            stem_l_max=None,  # len of l_max in stem (if using s4)
            downsample_type='patch',  # supports `s4nd` + avg pool
            downsample_act=False,
            downsample_glu=False,
            head_init_scale=1.,
            head_norm_first=False,
            norm_layer=None,
            custom_ln=False,
            drop_head=0.,
            drop_path_rate=0.,
            drop_mlp=0.,
            layer=None, # Shared config dictionary for the core layer
            stem_layer=None,
            stage_layers=None,
            video_size=None,
            block_tempor_kernel=3,  # only for non-s4 block
            temporal_stage_strides=None,
            factor_3d=False,
            **kwargs,  # catch all
    ):
        super().__init__()

        assert output_spatial_stride == 32
        if norm_layer is None:
            if custom_ln:
                norm_layer = TransposedLN
            else:
                norm_layer = partial(LayerNorm3d, eps=1e-6)
            cl_norm_layer = norm_layer if conv_mlp else partial(nn.LayerNorm, eps=1e-6)
        else:
            assert conv_mlp,\
                'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
            cl_norm_layer = norm_layer

        self.num_classes = num_classes
        self.drop_head = drop_head
        self.feature_info = []

        # Broadcast dictionaries
        if layer is not None:
            stage_layers = [OmegaConf.merge(layer, s) for s in stage_layers]
            stem_layer = OmegaConf.merge(layer, stem_layer)


        # instantiate stem here
        self.stem = Stem3d(
            stem_type=stem_type,  # supports `s4nd` + avg pool
            in_chans=in_chans,
            spatial_patch_size=spatial_patch_size,
            tempor_patch_size=tempor_patch_size,
            stem_channels=stem_channels,
            dims=dims,  
            stem_l_max=stem_l_max,  # len of l_max in stem (if using s4)
            norm_layer=norm_layer,
            custom_ln=custom_ln,
            layer=layer, # Shared config dictionary for the core layer
            stem_layer=stem_layer,
            factor_3d=factor_3d,
        )

        stem_stride = [tempor_patch_size, spatial_patch_size, spatial_patch_size]
        prev_chs = dims[0]

        # TODO: something else here?
        curr_video_size = [
            x // stem_stride if isinstance(stem_stride, int) else x // stem_stride[i]
            for i, x in enumerate(video_size)
        ]

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            # if stem downsampled by 4, then in stage 0, we don't downsample
            # if stem downsampled by 2, then in stage 0, we downsample by 2
            # all other stages we downsample by 2 no matter what
            # might want to alter the

            # temporal stride, we parse this specially
            tempor_stride = temporal_stage_strides[i] if temporal_stage_strides is not None else 2
            stride = [1, 1, 1] if i == 0 and np.any(np.array(stem_stride) >= 2) else [tempor_stride, 2, 2]  # stride 1 is no downsample (because already ds in stem)

            # print("stage {}, before downsampled img size {}, stride {}".format(i, curr_img_size, stride))
            out_chs = dims[i]
            stages.append(
                ConvNeXtStage3D(
                    prev_chs,
                    out_chs,
                    video_size=curr_video_size,
                    stride=stride,
                    depth=depths[i],
                    dp_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    norm_layer=norm_layer,
                    cl_norm_layer=cl_norm_layer,
                    stage_layer=stage_layers[i],
                    block_tempor_kernel=block_tempor_kernel,
                    downsample_type=downsample_type,
                    downsample_act=downsample_act,
                    downsample_glu=downsample_glu,
                    factor_3d=factor_3d,
                    drop_mlp=drop_mlp,
                )
            )

            prev_chs = out_chs
            # update image size for next stage
            curr_video_size = [
                x // stride if isinstance(stride, int) else x // stride[i]
                for i, x in enumerate(curr_video_size)
            ]

            # # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            # self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)

        self.num_features = prev_chs
        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
        self.head = nn.Sequential(OrderedDict([
                ('global_pool', nn.AdaptiveAvgPool3d(1)),
                ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
                ('flatten', nn.Flatten(1)),
                ('drop', nn.Dropout(self.drop_head)),
                ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))

        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm_pre', (99999,))
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, **kwargs):
        if global_pool is not None:
            self.head.global_pool = nn.AdaptiveAvgPool
            self.head.flatten = nn.Flatten(1)
        self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
        x = self.head.global_pool(x)
        x = self.head.norm(x)
        x = self.head.flatten(x)
        x = self.head.drop(x)
        return x if pre_logits else self.head.fc(x)

    def forward(self, x, state=None):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x, None

def _create_convnext3d(variant, pretrained=False, **kwargs):
    model = build_model_with_cfg(
        ConvNeXt3D,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


@register_model
def convnext3d_tiny(pretrained=False, **kwargs):
    model_args = dict(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), **kwargs)
    model = _create_convnext3d('convnext_tiny', pretrained=pretrained, **model_args)
    return model


def convnext_timm_tiny_2d_to_3d(model, state_dict, ignore_head=True, normalize=True):
    """
    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes (eg, imagenet to hmdb51), then you need to use this.
        normalize: bool, if set to True (default), it inflates with a factor of 1, and if
             set to False it inflates with a factor of 1/T where T is the temporal length for that kernel

    return:
        state_dict: dict, update with inflated weights
    """

    model_scratch_params_dict = dict(model.named_parameters())
    prefix = list(state_dict.keys())[0].split('.')[0]  # grab prefix in the keys for state_dict params

    old_state_dict = copy.deepcopy(state_dict)

    # loop through keys (in either)
    # only check `weights`
    # compare shapes btw 3d model and 2d model
    # if, different, then broadcast
    # then set the broadcasted version into the model value

    for key in sorted(model_scratch_params_dict.keys()):

        scratch_params = model_scratch_params_dict[key]

        # need to add the predix 'model' in convnext
        key_with_prefix = prefix + '.' + key

        # make sure key is in the loaded params first, if not, then print it out
        loaded_params = state_dict.get(key_with_prefix, None)

        if 'time_weight' in key:
            print("found time_weight parameter, train from scratch", key)
            used_params = scratch_params
        elif loaded_params is None:
            # This should never happen for 2D -> 3D ConvNext
            print("Missing key in pretrained model!", key_with_prefix)
            raise Exception
            # used_params = scratch_params

        elif ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, ignore", key)
            used_params = scratch_params

        elif len(scratch_params.shape) != len(loaded_params.shape):
            # same keys, but inflating weights
            print('key: shape DOES NOT MATCH', key)
            print("scratch:", scratch_params.shape)
            print("pretrain:", loaded_params.shape)
            # need the index [-3], 3rd from last, the temporal dim
            index = -3
            temporal_dim = scratch_params.shape[index]  # temporal len of kernel
            temporal_kernel_factor = 1 if normalize else 1 / temporal_dim
            used_params = repeat(temporal_kernel_factor*loaded_params, '... h w -> ... t h w', t=temporal_dim)
            # loaded_params = temporal_kernel_factor * loaded_params.unsqueeze(index)  # unsqueeze
            # used_params = torch.cat(temporal_dim * [loaded_params], axis=index)  # stack at this dim
        else:
            # print('key: shape MATCH', key)  # loading matched weights
            # used_params = loaded_params
            continue

        state_dict[key_with_prefix] = used_params

    return state_dict

def convnext_timm_tiny_s4nd_2d_to_3d(model, state_dict, ignore_head=True, jank=False):
    """
    inputs:
        model: nn.Module, the from 'scratch' model
        state_dict: dict, from the pretrained weights
        ignore_head: bool, whether to inflate weights in the head (or keep scratch weights).
            If number of classes changes (eg, imagenet to hmdb51), then you need to use this.

    return:
        state_dict: dict, update with inflated weights
    """

    # model_scratch_params_dict = dict(model.named_parameters())
    model_scratch_params_dict = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    prefix = list(state_dict.keys())[0].split('.')[0]  # grab prefix in the keys for state_dict params

    new_state_dict = copy.deepcopy(state_dict)

    # for key in state_dict.keys():
    #     print(key)

    # breakpoint()


    for key in sorted(model_scratch_params_dict.keys()):


        # need to add the predix 'model' in convnext
        key_with_prefix = prefix + '.' + key

        # HACK
        old_key_with_prefix = key_with_prefix.replace("inv_w_real", "log_w_real")

        # print(key)
        # if '.kernel.L' in key:
        #     print(key, state_dict[old_key_with_prefix])


        if '.kernel.0' in key:
            # temporal dim is loaded from scratch
            print("found .kernel.0:", key)
            new_state_dict[key_with_prefix] = model_scratch_params_dict[key]

        elif '.kernel.1' in key:
            # This is the 1st kernel --> 0th kernel from pretrained model
            print("FOUND .kernel.1, putting kernel 0 into kernel 1", key)
            new_state_dict[key_with_prefix] = state_dict[old_key_with_prefix.replace(".kernel.1", ".kernel.0")]
        elif '.kernel.2' in key:
            print("FOUND .kernel.2, putting kernel 1 into kernel 2", key)
            new_state_dict[key_with_prefix] = state_dict[old_key_with_prefix.replace(".kernel.2", ".kernel.1")]
        elif ignore_head and 'head' in key:
            # ignore head weights
            print("found head key / parameter, ignore", key)
            new_state_dict[key_with_prefix] = model_scratch_params_dict[key]
        # keys match
        else:
            # check if mismatched shape, if so, need to inflate
            # this covers cases where we did not use s4 (eg, optionally use conv2d in downsample or the stem)
            try:
                if model_scratch_params_dict[key].ndim != state_dict[old_key_with_prefix].ndim:
                    print("matching keys, but shapes mismatched!  Need to inflate!", key)
                    # need the index [-3], 3rd from last, the temporal dim
                    index = -3
                    dim_len = model_scratch_params_dict[key].shape[index]
                    # loaded_params = state_dict[key_with_prefix].unsqueeze(index)  # unsqueeze
                    # new_state_dict[key_with_prefix] = torch.cat(dim_len * [loaded_params], axis=index)  # stack at this dim
                    new_state_dict[key_with_prefix] = repeat(state_dict[old_key_with_prefix], '... h w -> ... t h w', t=dim_len) # torch.cat(dim_len * [loaded_params], axis=index)  # stack at this dim
                else:
                    # matching case, shapes, match, load into new_state_dict as is
                    new_state_dict[key_with_prefix] = state_dict[old_key_with_prefix]
            # something went wrong, the keys don't actually match (and they should)!
            except:
                print("unmatched key", key)
                breakpoint()
                # continue

    return new_state_dict

def main():
    model = convnext_tiny(
        stem_type='new_s4nd_patch',
        stem_channels=32,
        stem_l_max=[16, 16],
        downsample_type='s4nd',
        downsample_glu=True,
        stage_layers=[dict(dt_min=0.1, dt_max=1.0)] * 4,
        stem_layer=dict(dt_min=0.1, dt_max=1.0, init='fourier'),
        layer=dict(
            _name_='s4nd',
            bidirectional=True,
            init='fourier',
            dt_min=0.01,
            dt_max=1.0,
            n_ssm=1,
            return_state=False,
        ),
        img_size=[224, 224],
    )

    # model = convnext_tiny(
    #     stem_type='patch',
    #     downsample_type=None,
    #     stage_layers=[None] * 4,
    #     img_size=[224, 224],
    # )

    vmodel = convnext3d_tiny(
        stem_type='new_s4nd_patch',
        stem_channels=32,
        stem_l_max=[100, 16, 16],
        downsample_type='s4nd',
        downsample_glu=True,
        stage_layers=[dict(dt_min=0.1, dt_max=1.0)] * 4,
        stem_layer=dict(dt_min=0.1, dt_max=1.0, init='fourier'),
        layer=dict(
            _name_='s4nd',
            bidirectional=True,
            init='fourier',
            dt_min=0.01,
            dt_max=1.0,
            n_ssm=1,
            contract_version=1,
            return_state=False,
        ),
        video_size=[100, 224, 224],
    )

    # vmodel = convnext3d_tiny(
    #     stem_type='patch',
    #     downsample_type=None,
    #     stage_layers=[None] * 4,
    #     video_size=[100, 224, 224],
    # )
    model.cuda()
    x = torch.rand(1, 3, 224, 224).cuda()
    y = model(x)[0]
    print(y)
    breakpoint()

    vmodel.cuda()
    x = torch.rand(1, 3, 50, 224, 224).cuda()
    y = vmodel(x)[0]
    print(y)
    print(y.shape)

    breakpoint()

    # 3D Stem Conv options
    # 1, 4, 4 kernel and stride
    # 7, 4, 4 kernel and stride 2, 4, 4


# =======================================
# s4/configs/experiment/s4nd/convnext/convnext_timm_tiny_s4nd_imagenet.yaml
"""
model:
  img_size: ${dataset.__l_max}
  drop_path_rate: 0.1
  patch_size: 4  # 2 or 4, use for stem downsample factor
  stem_channels: 32  # only used for s4nd stem currently
  stem_type: new_s4nd_patch  # options: patch (regular convnext), s4nd_patch, new_s4nd_patch (best), s4nd
  stem_l_max: [16, 16]  # stem_l_max=None,  # len of l_max in stem (if using s4)
  downsample_type: s4nd  # eg, s4nd, null (for regular strided conv)
  downsample_act: false
  downsample_glu: True
  conv_mlp: false
  custom_ln: false # only used if conv_mlp=1, should benchmark to make sure this is faster/more mem efficient, also need to turn off weight decay
  layer:  # null means use regular conv2d in convnext
    _name_: s4nd
    d_state: 64
    channels: 1
    bidirectional: true
    activation: null  # mimics convnext style
    final_act: none
    initializer: null
    weight_norm: false
    dropout: 0
    tie_dropout: ${oc.select:model.tie_dropout,null}
    init: fourier
    rank: 1
    trank: 1
    dt_min: 0.01
    dt_max: 1.0
    lr: 0.001
    # length_correction: true
    n_ssm: 1
    deterministic: false # Special C init
    l_max: ${oc.select:dataset.__l_max,null} # Grab dataset length if exists, otherwise set to null and kernel will automatically resize
    verbose: true
    linear: true
    return_state: false
    bandlimit: null
    contract_version: 0  # 0 is for 2d, 1 for 1d or 3d (or other)
  stem_layer:
    dt_min: 0.1
    dt_max: 1.0
    init: fourier
  stage_layers:
    - dt_min: 0.1
      dt_max: 1.0
    - dt_min: 0.1
      dt_max: 1.0
    - dt_min: 0.1
      dt_max: 1.0
    - dt_min: 0.1
      dt_max: 1.0
"""

# s4/configs/model/layer/s4nd.yaml
"""
_name_: s4nd
d_state: 64
channels: 1
bidirectional: true
activation: gelu
final_act: glu
initializer: null
weight_norm: false
trank: 1
dropout: ${..dropout} # Same as null
tie_dropout: ${oc.select:model.tie_dropout,null}
init: legs
rank: 1
dt_min: 0.001
dt_max: 0.1
lr:
  dt: 0.001
  A: 0.001
  B: 0.001
n_ssm: 1
deterministic: false # Special C init
l_max: ${oc.select:dataset.__l_max,null} # Grab dataset length if exists, otherwise set to null and kernel will automatically resize
verbose: true
linear: false
"""

def convnext_tiny_s4nd():
    model = ConvNeXt(
        depths=(3, 3, 9, 3), 
        dims=(96, 192, 384, 768),
        img_size=(224, 224),
        patch_size=4,
        stem_channels=32,
        stem_type="new_s4nd_patch",
        stem_l_max=[16, 16],
        downsample_act=False,
        downsample_glu=True,
        conv_mlp=False,
        custom_ln=False,
        layer=dict(
            _name_="s4nd",
            d_state=64,
            channels=1,
            bidirectional=True,
            # activation="null",
            # final_act="none",
            final_act=None,
            # initializer="null",
            weight_norm=False,
            dropout=0,
            # tie_dropout="null",
            init="fourier",
            rank=1,
            trank=1,
            dt_min=0.01,
            dt_max=1.0,
            lr=0.001,
            n_ssm=1,
            deterministic=False,
            # l_max="null",
            verbose=True,
            linear=True,
            return_state=False,
            # bandlimit="null",
            contract_version=0,
        ),
        stem_layer=dict(
            dt_min=0.1, 
            dt_max=1.0, 
            init="fourier",
        ),
        stage_layers=[
            dict(dt_min=0.1, dt_max=1.0),
            dict(dt_min=0.1, dt_max=1.0),
            dict(dt_min=0.1, dt_max=1.0),
            dict(dt_min=0.1, dt_max=1.0),
        ],
    )
    
    def forward(self, x, resolution=1, state=None):
        x = self.forward_features(x, resolution)
        x = self.forward_head(x)
        return x
    
    model.forward = partial(forward, model)
    return model

if __name__ == '__main__':
    convnext_tiny_s4nd()


