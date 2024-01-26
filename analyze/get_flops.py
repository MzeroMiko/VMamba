# Analyze flops ================================
from typing import Callable, Tuple, Union, Tuple, Union, Any
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

from mmengine.analysis.print_helper import is_tuple_of, FlopAnalyzer, ActivationAnalyzer, parameter_count, _format_size, complexity_stats_table, complexity_stats_str
from mmengine.analysis.jit_analysis import _IGNORED_OPS
from mmengine.analysis.complexity_analysis import _DEFAULT_SUPPORTED_FLOP_OPS, _DEFAULT_SUPPORTED_ACT_OPS
from mmengine.analysis import get_model_complexity_info as mm_get_model_complexity_info

from vmamba.vmamba import VSSM, selective_scan_flop_jit
from vmamba.vss import VSS
# from vmamba.vssmd import VSSMD as VSSM


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
    """Interface to get the complexity of a model.

    The parameter `inputs` are fed to the forward method of model.
    If `inputs` is not specified, the `input_shape` is required and
    it will be used to construct the dummy input fed to model.
    If the forward of model requires two or more inputs, the `inputs`
    should be a tuple of tensor or the `input_shape` should be a tuple
    of tuple which each element will be constructed into a dumpy input.

    Examples:
        >>> # the forward of model accepts only one input
        >>> input_shape = (3, 224, 224)
        >>> get_model_complexity_info(model, input_shape=input_shape)
        >>> # the forward of model accepts two or more inputs
        >>> input_shape = ((3, 224, 224), (3, 10))
        >>> get_model_complexity_info(model, input_shape=input_shape)

    Args:
        model (nn.Module): The model to analyze.
        input_shape (Union[Tuple[int, ...], Tuple[Tuple[int, ...]], None]):
            The input shape of the model.
            If "inputs" is not specified, the "input_shape" should be set.
            Defaults to None.
        inputs (torch.Tensor, tuple[torch.Tensor, ...] or Tuple[Any, ...],\
            optional]):
            The input tensor(s) of the model. If not given the input tensor
            will be generated automatically with the given input_shape.
            Defaults to None.
        show_table (bool): Whether to show the complexity table.
            Defaults to True.
        show_arch (bool): Whether to show the complexity arch.
            Defaults to True.

    Returns:
        dict: The complexity information of the model.
    """
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


    supported_ops={
        "aten::silu": None, # as relu is in _IGNORED_OPS
        "aten::neg": None, # as relu is in _IGNORED_OPS
        "aten::exp": None, # as relu is in _IGNORED_OPS
        "aten::flip": None, # as permute is in _IGNORED_OPS
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
    }
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


def mmengine_flop_count(model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False,):
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


def fvcore_flop_count(model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False, show_arch=False):
    from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
    from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
    from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
    from fvcore.nn.jit_analysis import _IGNORED_OPS
    from fvcore.nn.jit_handles import get_shape, addmm_flop_jit
    
    supported_ops={
        "aten::silu": None, # as relu is in _IGNORED_OPS
        "aten::neg": None, # as relu is in _IGNORED_OPS
        "aten::exp": None, # as relu is in _IGNORED_OPS
        "aten::flip": None, # as permute is in _IGNORED_OPS
        "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
    }

    class _FlopCountAnalysis(FlopCountAnalysis):
        def __init__(self, model: Module, inputs: Tensor | Tuple[Tensor, ...]) -> None:
            super().__init__(model, inputs)
            self.set_op_handle(**supported_ops)
    
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
        flops = _FlopCountAnalysis(model, inputs),
        max_depth=100,
        activations=None,
        show_param_shapes=True,
    )

    flops_str = flop_count_str(
        flops = _FlopCountAnalysis(model, inputs),
        activations=None,
    )

    if show_arch:
        print(flops_str)

    if show_table:
        print(flops_table)
    
    print(Gflops.items())

    params = fvcore_parameter_count(model)[""]
    flops = sum(Gflops.values())
    print("GFlops: ", flops, "Params: ", params, flush=True)
    return params, flops

# ==============================

def build_models(model="vssm"):
    if model in ["vssm"]:
        model = VSSM(depths=[2, 2, 9, 2], dims=96)
        def forward_backbone(self: VSSM, x):
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)
            for layer in self.layers:
                x = layer(x)
            return x

        model.forward = partial(forward_backbone, model)
        del model.norm
        del model.avgpool
        del model.head
        model.cuda().eval()
        return model
    elif model in ["swin"]:
        from mmpretrain.models.backbones import SwinTransformer
        model = SwinTransformer(arch="tiny")
        [setattr(model, f"norm{i}", nn.Identity()) for i in model.out_indices]
        model.cuda().eval()
        return model
    elif model in ["convnext"]:
        from mmpretrain.models.backbones import ConvNeXt
        model = ConvNeXt(arch="tiny")
        model.gap_before_final_norm = False
        [setattr(model, f"norm{i}", nn.Identity()) for i in model.out_indices]
        model.cuda().eval()
        return model
    elif model in ["replknet"]:
        # norm here is hard to delete
        from mmpretrain.models.backbones import RepLKNet
        model = RepLKNet(arch="31B")
        model.cuda().eval()
        return model
    elif model in ["deit"]:
        from mmpretrain.models.backbones import VisionTransformer
        model = VisionTransformer(arch="deit-small")
        model.out_type = 'featmap'
        model.cuda().eval()
        return model
    elif model in ["resnet50"]:
        from mmpretrain.models.backbones import ResNet
        model = ResNet(depth=50)
        model.cuda().eval()
        return model
    else:
        raise NotImplementedError


def build_model_vssm(depths=[2, 2, 9, 2], embed_dim=96):
    model = VSSM(depths=depths, dims=embed_dim, drop_path_rate=0.2)
    def forward_backbone(self: VSSM, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        return x

    model.forward = partial(forward_backbone, model)
    del model.norm
    del model.avgpool
    del model.head
    model.cuda().eval()
    return model
    

def get_flops_vssm(core="fvcore"):
    """
    convnext: bs4096
          # 224 tiny
          FLOPs: 4457472768 Parameters: 28589128 FLOPs: 13135236864
          Top 1 Accuracy: 82.14
          Top 1 Accuracy: 81.95 # noema
          Top 1 Accuracy: 82.90 # 21kpre
          Top 1 Accuracy: 84.11 # 21kpre + 384

          # 224 small
          FLOPs: 8687008512 Parameters: 50223688 FLOPs: 25580818176
          Top 1 Accuracy: 83.16
          Top 1 Accuracy: 83.21 # noema
          Top 1 Accuracy: 84.59 # 21kpre
          Top 1 Accuracy: 85.75 # 21kpre + 384

          # 224 base
          FLOPs: 15359124480 Parameters: 88591464 FLOPs: 45205885952
          Top 1 Accuracy: 83.66
          Top 1 Accuracy: 83.64 # noema
          Top 1 Accuracy: 85.81 # 21kpre
          Top 1 Accuracy: 86.82 # 21kpre + 384
    swin: bs1024
          # 224 tiny
          FLOPs: 4360000000 Parameters: 28290000
          Top 1 Accuracy: 81.18

          # 224 small
          FLOPs: 8520000000 Parameters: 49610000
          Top 1 Accuracy: 83.02

          # 224 base
          FLOPs: 15140000000 Parameters: 87770000 FLOPs: 44490000000
          Top 1 Accuracy: 83.36
          Top 1 Accuracy: 84.49 # 384
          Top 1 Accuracy: 85.16 # 21kpre
          Top 1 Accuracy: 86.44 # 21kpre, 384
    """

    _flops_count = fvcore_flop_count
    if core.startswith("mm"):
        _flops_count = mmengine_flop_count
    build_vmamba = build_model_vssm

    _flops_count(build_vmamba(depths=[2, 2, 18, 2], embed_dim=128), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 18, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 6, 2], embed_dim=96), input_shape=(3, 224, 224))
    # for ssmd: 15.7 + 85.7; 9.06 + 48.7; 4.72 + 27.3;

    _flops_count(build_vmamba(depths=[8, 8, 30, 8], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[6, 6, 36, 6], embed_dim=96), input_shape=(3, 224, 224))
    # 15.7 + 76.6; 15.3 + 74.0;


    _flops_count(build_vmamba(depths=[2, 3, 9, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 3, 27, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 3, 27, 2], embed_dim=128), input_shape=(3, 224, 224))
    # 4.77 + 22.5, 9.42 + 44.0, 15.70 + 75.8

    _flops_count(build_vmamba(depths=[3, 3, 6, 3], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[3, 3, 24, 3], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[3, 3, 24, 3], embed_dim=128), input_shape=(3, 224, 224))
    # 4.66 + 23.4, 9.3 + 44.9, 15.45 + 77.6

    _flops_count(build_vmamba(depths=[2, 2, 9, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 27, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 27, 2], embed_dim=128), input_shape=(3, 224, 224))
    # 4.46 + 22.1, 9.11 + 43.6, 15.2 + 75.2


def get_scale_up():
    _flops_count = fvcore_flop_count
    for model in ["vssm", "swin", "convnext", "replknet", "deit", "resnet50"]:
        print(f"FLOPs for model {model} with different input shapes ==")
        build_model = lambda *args: build_models(model)
        for input_shape in [64, 112, 224, 384, 512, 640, 768, 1024, 1120, 1280]:
            try:
                _model = build_model()
                params, gflops = _flops_count(_model, input_shape=(3, input_shape, input_shape))
                print(f"model {model} + input shape {input_shape} => params {params} GFLOPs {gflops}", flush=True)
            except Exception as e:
                print(e)


def mmdet_mmseg_vssm(FORCE_BUILD=True):
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
            if not FORCE_BUILD:
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
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

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


def mmseg_flops(config=None, input_shape=(3, 512, 2048)):
    from mmengine.config import Config, DictAction
    from mmengine.runner import Runner
    from mmseg.registry import RUNNERS
    from mmseg.models.data_preprocessor import SegDataPreProcessor
    from mmseg.models.builder import MODELS
    import numpy as np
    import os

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    
    fvcore_flop_count(model, input_shape=input_shape)


def mmdet_flops(config=None):
    from mmengine.config import Config, DictAction
    from mmengine.runner import Runner
    from mmseg.registry import RUNNERS
    from mmseg.models.data_preprocessor import SegDataPreProcessor
    from mmseg.models.builder import MODELS
    import numpy as np
    import os

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    
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

    
if __name__ == '__main__':
    if True:
        print("fvcore flops count for vssm ====================", flush=True)
        get_flops_vssm()
        print("mmengine flops count for vssm ====================", flush=True)
        get_flops_vssm("mm") # same as fvcore
        print("flops count for models with bigger inputs ====================", flush=True)
        get_scale_up()  

    if True:
        mmdet_mmseg_vssm()
        segpath = "segmentation/configs"
        detpath = "detection/configs"
        if True:
            mmseg_flops(config=f"{segpath}/upernet/upernet_r50_4xb4-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  952.616667136 Params:  66516108
            mmseg_flops(config=f"{segpath}/upernet/upernet_r101_4xb4-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  1030.4084234239997 Params:  85508236
            mmseg_flops(config=f"{segpath}/vit/vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  1216.821829632 Params:  57994796
            mmseg_flops(config=f"{segpath}/vit/vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py", input_shape=(3, 512, 2048)) # GFlops:  2006.545496064 Params:  144172844
            mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py", input_shape=(3, 512, 2048)) # GFlops:  939.4933174400002 Params:  54546956
            mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py", input_shape=(3, 512, 2048)) # GFlops:  1036.6845167359998 Params:  76070924
            mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py", input_shape=(3, 512, 2048)) # GFlops:  1166.887735664 Params:  109765548
            mmseg_flops(config=f"{segpath}/vssm/upernet_swin_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1614.082896384 Params:  81259766
            mmseg_flops(config=f"{segpath}/vssm/upernet_convnext_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1606.538496 Params:  81877196
            mmseg_flops(config=f"{segpath}/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1619.8110944 Params:  76070924
    
        if True:
            mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_swin_fpn_coco_tiny.py") # 42.4M 262093532640.0
            mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_swin_fpn_coco_small.py") # 63.924M 357006236640.0
            mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_swin_fpn_coco_base.py") # 95.628M 482127568640.0
            mmdet_flops(config=f"{detpath}/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py") # 44.396M 260152304640.0
            mmdet_flops(config=f"{detpath}/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py") # 63.388M 336434160640.0

    
