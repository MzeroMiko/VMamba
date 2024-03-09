import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module
from functools import partial
from typing import Callable, Tuple, Union, Tuple, Union, Any

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
selective_scan_flop_jit: Callable = build.vmamba.selective_scan_flop_jit
VSSM: nn.Module = build.vmamba.VSSM
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM

supported_ops={
    "aten::silu": None, # as relu is in _IGNORED_OPS
    "aten::neg": None, # as relu is in _IGNORED_OPS
    "aten::exp": None, # as relu is in _IGNORED_OPS
    "aten::flip": None, # as permute is in _IGNORED_OPS
    "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit, # latter
    "prim::PythonOp.SelectiveScan": selective_scan_flop_jit, # latter
}

def mmengine_flop_count(model: nn.Module = None, input_shape = (3, 224, 224), show_table=False, show_arch=False, _get_model_complexity_info=False):
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


def fvcore_flop_count(model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False, show_arch=False):
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
    
    print(Gflops.items())

    params = fvcore_parameter_count(model)[""]
    flops = sum(Gflops.values())
    print("GFlops: ", flops, "Params: ", params, flush=True)
    return params, flops


# ==============================


def build_model_vssm(depths=[2, 2, 9, 2], embed_dim=96):
    model = VSSM(depths=depths, dims=embed_dim, d_state=16, dt_rank="auto", ssm_ratio=2.0, mlp_ratio=0.0, downsample="v1")
    def forward_backbone(self: VSSM, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x

    model.forward = partial(forward_backbone, model)
    try:
        del model.norm
    except:
        pass
    try:
        del model.head
    except:
        pass
    try:
        del model.classifier
    except:
        pass
    model.cuda().eval()
    return model
    

def vssm_flops(core="fvcore"):
    _flops_count = fvcore_flop_count
    if core.startswith("mm"):
        _flops_count = mmengine_flop_count
    build_vmamba = build_model_vssm
    _flops_count(build_vmamba(depths=[2, 2, 9, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 27, 2], embed_dim=96), input_shape=(3, 224, 224))
    _flops_count(build_vmamba(depths=[2, 2, 27, 2], embed_dim=128), input_shape=(3, 224, 224))
    # 4.46 + 22.1, 9.11 + 43.6, 15.2 + 75.2


def mmdet_mmseg_vssm():
    from mmengine.model import BaseModule
    from mmdet.registry import MODELS as MODELS_MMDET
    from mmseg.registry import MODELS as MODELS_MMSEG

    @MODELS_MMSEG.register_module()
    @MODELS_MMDET.register_module()
    class MM_VSSM(BaseModule, Backbone_VSSM):
        def __init__(self, *args, **kwargs):
            BaseModule.__init__(self)
            Backbone_VSSM.__init__(self, *args, **kwargs)


def mmseg_flops(config=None, input_shape=(3, 512, 2048)):
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    
    fvcore_flop_count(model, input_shape=input_shape)


def mmdet_flops(config=None):
    from mmengine.config import Config
    from mmengine.runner import Runner
    import numpy as np
    import os

    cfg = Config.fromfile(config)
    cfg["work_dir"] = "/tmp"
    runner = Runner.from_cfg(cfg)
    model = runner.model.cuda()
    get_model_complexity_info = mmengine_flop_count(_get_model_complexity_info=True)
    
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
    if False:
        print("fvcore flops count for vssm ====================", flush=True)
        vssm_flops()
        print("mmengine flops count for vssm ====================", flush=True)
        vssm_flops("mm") # same as fvcore

    segpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../segmentation/configs")
    detpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../detection/configs")

    if True:
        mmdet_mmseg_vssm()
        if False:
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
    
        if False:
            mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_vssm_fpn_coco_tiny.py") # 42.4M 262093532640.0
            mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_vssm_fpn_coco_small.py") # 63.924M 357006236640.0
            mmdet_flops(config=f"{detpath}/vssm/mask_rcnn_vssm_fpn_coco_base.py") # 95.628M 482127568640.0
            mmdet_flops(config=f"{detpath}/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py") # 44.396M 260152304640.0
            mmdet_flops(config=f"{detpath}/mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py") # 63.388M 336434160640.0

        if True:
            mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py", input_shape=(3, 512, 2048)) # GFlops:  947.7798358240001 Params:  62359340
            mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py", input_shape=(3, 512, 2048)) # GFlops:  1028.404888464 Params:  81801260
            mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py", input_shape=(3, 512, 2048)) # GFlops:  1170.3442882240001 Params:  122069292
            mmseg_flops(config=f"{segpath}/vssm1/upernet_vssm_4xb4-160k_ade20k-640x640_small.py", input_shape=(3, 640, 2560)) # GFlops:  1606.8682596 Params:  81801260
    
        if True:
            mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py") # 50.212M 270186348640.0
            mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_small.py") # 69.654M 348921708640.0
            mmdet_flops(config=f"{detpath}/vssm1/mask_rcnn_vssm_fpn_coco_base.py") # 0.108G 485496108640.0

    
