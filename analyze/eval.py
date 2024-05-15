import time
import tqdm
import torch
import torch.utils.data
import argparse
import os
import sys
import logging
from functools import partial
from torchvision import datasets, transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.models.vision_transformer import EncoderBlock
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import math
logging.basicConfig(level=logging.INFO)
logger = logging
from timm.utils import accuracy, AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


HOME = os.environ["HOME"].rstrip("/")
basicpath = os.path.abspath("../VMamba/analyze").rstrip("/")
basicpath = os.path.abspath(os.path.dirname(__file__)).rstrip("/")

# this mode will greatly inference the speed!
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


# copied from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


# WARNING!!!  acc score would be inaccurate if num_procs > 1, as sampler always pads the dataset
# copied from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


def get_dataloader(batch_size=64, root="./val", img_size=224, sequential=True):
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    if sequential:
        sampler = torch.utils.data.SequentialSampler(dataset)
    else:
        sampler = torch.utils.data.DistributedSampler(dataset)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


def _validate(
    model: nn.Module = None, 
    freq=10, 
    amp=True, 
    img_size=224, 
    batch_size=128, 
    data_path="/dataset/ImageNet2012",
):
    class Args():
        AMP_ENABLE = amp
        PRINT_FREQ = freq
    config = Args()

    model.cuda().eval()
    model = torch.nn.parallel.DistributedDataParallel(model)
    _batch_size = batch_size
    while _batch_size > 0:
        try:
            _dataloader = get_dataloader(
                batch_size=_batch_size, 
                root=os.path.join(os.path.abspath(data_path), "val"),
                img_size=img_size,
                sequential=False,
            )
            logging.info(f"starting loop: img_size {img_size}; len(dataset) {len(_dataloader.dataset)}")
            validate(config, data_loader=_dataloader, model=model)
            break
        except:
            _batch_size = _batch_size // 2
            print(f"batch_size {_batch_size}", flush=True)


def testscale(modes, batch_size=32, data_path="ImageNet_ILSVRC2012"):
    sizes = [224, 384, 512, 640, 768, 1024]
    _build = import_abspy("models", f"{os.path.dirname(__file__)}/../classification")
    build_mmpretrain_models = _build.build_mmpretrain_models

    if "vssm1" in modes:
        print("vssm taav1 ================================", flush=True)
        import triton, mamba_ssm, selective_scan_cuda_oflex
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        taav1 = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth"
        model = taav1().cuda().eval()
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        for size in sizes:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

    if "vssm0" in modes:
        print("vssm ta6 ================================", flush=True)
        import triton, mamba_ssm, selective_scan_cuda_oflex
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        ta6 = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
        model = ta6().cuda().eval()
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        for size in sizes:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

    if "resnet" in modes:
        print("resnet ================================", flush=True)
        model = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=False, with_norm=True,)
        model = model()
        for size in [224, 384, 512, 640, 768, 1024]:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

    if "deit" in modes:
        print("deit ================================", flush=True)
        model = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=False, with_norm=True,)
        model = model()
        for size in [224, 384, 512, 640, 768, 1024]:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

    if "convnext" in modes:
        print("convnext ================================", flush=True)
        _model = import_abspy("convnext", f"{HOME}/OTHERS/ConvNeXt/models")
        model = _model.convnext_tiny()
        ckpt ="/home/LiuYue/Workspace/PylanceAware/ckpts/others/convnext_tiny_1k_224_ema.pth"
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        for size in [224, 384, 512, 640, 768, 1024]:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

    if "swinscale" in modes:
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
        print("swin ================================", flush=True)
        model = dict(
            type='ImageClassifier',
            backbone=dict(
                type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=768,
                init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
        )
        ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth"
        # for size in [224, 384, 512, 640, 768, 1024]:
        for size in [384, 512, 640, 768, 1024]:
            model["backbone"].update({"window_size": int(size // 32)})
            tiny = build_classifier(model)
            tiny.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
            _validate(tiny, img_size=size, batch_size=batch_size, data_path=data_path)

    if "hivit" in modes:
        from mmpretrain.models.builder import MODELS
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, HiViT, VisionTransformer, SwinTransformer
        from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np
        
        @MODELS.register_module()
        class HiViTx(HiViT):
            def __init__(self, *args,**kwargs):
                super().__init__(*args,**kwargs)
                self.num_extra_tokens = 0
                self.interpolate_mode = "bicubic"
                self.patch_embed.init_out_size = self.patch_embed.patches_resolution
                self._register_load_state_dict_pre_hook(self._prepare_abs_pos_embed)
                self._register_load_state_dict_pre_hook(
                    self._prepare_relative_position_bias_table)

            # copied from SwinTransformer, change absolute_pos_embed to pos_embed
            def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
                name = prefix + 'pos_embed'
                if name not in state_dict.keys():
                    return

                ckpt_pos_embed_shape = state_dict[name].shape
                if self.pos_embed.shape != ckpt_pos_embed_shape:
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(
                        'Resize the pos_embed shape from '
                        f'{ckpt_pos_embed_shape} to {self.pos_embed.shape}.')

                    ckpt_pos_embed_shape = to_2tuple(
                        int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
                    pos_embed_shape = self.patch_embed.init_out_size

                    state_dict[name] = resize_pos_embed(state_dict[name],
                                                        ckpt_pos_embed_shape,
                                                        pos_embed_shape,
                                                        self.interpolate_mode,
                                                        self.num_extra_tokens)

            def _prepare_relative_position_bias_table(self, state_dict, *args, **kwargs):
                del state_dict['backbone.relative_position_index']
                return SwinTransformer._prepare_relative_position_bias_table(self, state_dict, *args, **kwargs)

        print("hivit ================================", flush=True)
        model = dict(
            backbone=dict(
                ape=True,
                arch='tiny',
                drop_path_rate=0.05,
                img_size=224,
                rpe=True,
                type='HiViTx'),
            head=dict(
                cal_acc=False,
                in_channels=384,
                init_cfg=None,
                loss=dict(
                    label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
                num_classes=1000,
                type='LinearClsHead'),
            init_cfg=[
                dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
                dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
            ],
            neck=dict(type='GlobalAveragePooling'),
            train_cfg=dict(augments=[
                dict(alpha=0.8, type='Mixup'),
                dict(alpha=1.0, type='CutMix'),
            ]),
            type='ImageClassifier')
        ckpt="/home/LiuYue/Workspace/PylanceAware/ckpts/others/hivit-tiny-p16_8xb128_in1k/epoch_295.pth"
        for size in [224, 384, 512, 640, 768, 1024]:
        # for size in [384, 512, 640, 768, 1024]:
            model["backbone"].update({"img_size": size})
            tiny = build_classifier(model)
            tiny.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
            _validate(tiny, img_size=size, batch_size=batch_size, data_path=data_path)

    if "swin" in modes:
        print("swin ================================", flush=True)
        model = partial(build_mmpretrain_models, cfg="swin_tiny", ckpt=True, only_backbone=False, with_norm=True,)
        model = model()
        for size in [224, 384, 512, 640, 768, 1024]:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

    if "intern" in modes:
        print("intern ================================", flush=True)
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        model = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        model = model()
        ckpt =f"{HOME}/Workspace/PylanceAware/ckpts/others/internimage_t_1k_224.pth"
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        for size in [224, 384, 512, 640, 768, 1024]:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)
        sys.path = sys.path[1:]

    if "xcit" in modes:
        xcit = import_abspy("xcit", f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/")
        model = xcit.xcit_small_12_p16()
        ckpt =f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/others/xcit_small_12_p16_224.pth"
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        for size in [224, 384, 512, 640, 768, 1024]:
            _validate(model, img_size=size, batch_size=batch_size, data_path=data_path)

def testperf(modes, batch_size=128, data_path="ImageNet_ILSVRC2012"):
    _build = import_abspy("models", f"{os.path.dirname(__file__)}/../classification")
    build_mmpretrain_models = _build.build_mmpretrain_models

    # ok
    if "vssm1" in modes:
        print("vssm1 ================================", flush=True)
        import triton, mamba_ssm, selective_scan_cuda_oflex
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        t0230s = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
        s0229 = partial(_model.VSSM, dims=96, depths=[2,2,15,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
        b0229 = partial(_model.VSSM, dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
        tckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth"
        sckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_small_0229/vssm1_small_0229_ckpt_epoch_222.pth"
        bckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_base_0229/vssm1_base_0229_ckpt_epoch_237.pth"
        model = t0230s().cuda().eval()
        model.load_state_dict(torch.load(open(tckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)
        model = s0229().cuda().eval()
        model.load_state_dict(torch.load(open(sckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)
        model = b0229().cuda().eval()
        model.load_state_dict(torch.load(open(bckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)

    # ok
    if "vssm0" in modes:
        print("vssm0 ================================", flush=True)
        import triton, mamba_ssm, selective_scan_cuda_oflex
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        ta6 = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
        sa6 = partial(_model.VSSM, dims=96, depths=[2,2,27,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
        ba6 = partial(_model.VSSM, dims=128, depths=[2,2,27,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
        tckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
        sckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmsmall/vssmsmall_dp03_ckpt_epoch_238.pth"
        bckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmbase/vssmbase_dp06_ckpt_epoch_241.pth"
        model = ta6().cuda().eval()
        model.load_state_dict(torch.load(open(tckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)
        model = sa6().cuda().eval()
        model.load_state_dict(torch.load(open(sckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)
        model = ba6().cuda().eval()
        model.load_state_dict(torch.load(open(bckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)

    # ok
    if "resnet" in modes:
        print("resnet ================================", flush=True)
        model = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=False, with_norm=True,)
        model = model()
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)

    # ok
    if "deit" in modes:
        print("deit ================================", flush=True)
        model = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=False, with_norm=True,)
        model = model()
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)

    # ok
    if "convnext" in modes:
        print("convnext ================================", flush=True)
        _model = import_abspy("convnext", f"{HOME}/OTHERS/ConvNeXt/models")
        model = _model.convnext_tiny()
        ckpt ="/home/LiuYue/Workspace/PylanceAware/ckpts/others/convnext_tiny_1k_224_ema.pth"
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)

    # ok
    if "swinscale" in modes:
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
        print("swin ================================", flush=True)
        model = dict(
            type='ImageClassifier',
            backbone=dict(
                type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=1000,
                in_channels=768,
                init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                loss=dict(
                    type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='Mixup', alpha=0.8),
                dict(type='CutMix', alpha=1.0)
            ]),
        )
        ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth"
        model["backbone"].update({"window_size": int(size // 32)})
        tiny = build_classifier(model)
        tiny.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        _validate(tiny, img_size=224, batch_size=batch_size, data_path=data_path)

    if "hivit" in modes:
        from mmpretrain.models.builder import MODELS
        from mmengine.runner import CheckpointLoader
        from mmpretrain.models import build_classifier, ImageClassifier, HiViT, VisionTransformer, SwinTransformer
        from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np
        
        @MODELS.register_module()
        class HiViTx(HiViT):
            def __init__(self, *args,**kwargs):
                super().__init__(*args,**kwargs)
                self.num_extra_tokens = 0
                self.interpolate_mode = "bicubic"
                self.patch_embed.init_out_size = self.patch_embed.patches_resolution
                self._register_load_state_dict_pre_hook(self._prepare_abs_pos_embed)
                self._register_load_state_dict_pre_hook(
                    self._prepare_relative_position_bias_table)

            # copied from SwinTransformer, change absolute_pos_embed to pos_embed
            def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
                name = prefix + 'pos_embed'
                if name not in state_dict.keys():
                    return

                ckpt_pos_embed_shape = state_dict[name].shape
                if self.pos_embed.shape != ckpt_pos_embed_shape:
                    from mmengine.logging import MMLogger
                    logger = MMLogger.get_current_instance()
                    logger.info(
                        'Resize the pos_embed shape from '
                        f'{ckpt_pos_embed_shape} to {self.pos_embed.shape}.')

                    ckpt_pos_embed_shape = to_2tuple(
                        int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
                    pos_embed_shape = self.patch_embed.init_out_size

                    state_dict[name] = resize_pos_embed(state_dict[name],
                                                        ckpt_pos_embed_shape,
                                                        pos_embed_shape,
                                                        self.interpolate_mode,
                                                        self.num_extra_tokens)

            def _prepare_relative_position_bias_table(self, state_dict, *args, **kwargs):
                del state_dict['backbone.relative_position_index']
                return SwinTransformer._prepare_relative_position_bias_table(self, state_dict, *args, **kwargs)

        print("hivit ================================", flush=True)
        size = 224
        model = dict(
            backbone=dict(
                ape=True,
                arch='tiny',
                drop_path_rate=0.05,
                img_size=224,
                rpe=True,
                type='HiViTx'),
            head=dict(
                cal_acc=False,
                in_channels=384,
                init_cfg=None,
                loss=dict(
                    label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
                num_classes=1000,
                type='LinearClsHead'),
            init_cfg=[
                dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
                dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
            ],
            neck=dict(type='GlobalAveragePooling'),
            train_cfg=dict(augments=[
                dict(alpha=0.8, type='Mixup'),
                dict(alpha=1.0, type='CutMix'),
            ]),
            type='ImageClassifier')
        ckpt="/home/LiuYue/Workspace/PylanceAware/ckpts/others/hivit-tiny-p16_8xb128_in1k/epoch_295.pth"
        model["backbone"].update({"img_size": size})
        tiny = build_classifier(model)
        tiny.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        _validate(tiny, img_size=size, batch_size=batch_size, data_path=data_path)

    if "intern" in modes:
        print("intern ================================", flush=True)
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        model = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        model = model()
        ckpt ="/home/LiuYue/Workspace/PylanceAware/ckpts/others/internimage_t_1k_224.pth"
        model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
        _validate(model, img_size=224, batch_size=batch_size, data_path=data_path)
        sys.path = sys.path[1:]

from utils import ExtractFeatures, BuildModels

extract_feature = ExtractFeatures.extract_feature


def _extract_feature(data_path="ImageNet_ILSVRC2012", start=0, end=200, step=-1, img_size=224, batch_size=16, train=True, aug=False):
    _build = import_abspy("models", f"{os.path.dirname(__file__)}/../classification")
    build_mmpretrain_models = _build.build_mmpretrain_models

    if True:
        resnet50 = BuildModels.build_resnet_mmpretrain(with_ckpt=True, remove_head=True, scale="r50", size=img_size).cuda().eval()
        deitsmall = BuildModels.build_deit_mmpretrain(with_ckpt=True, remove_head=True, scale="small", size=img_size).cuda().eval()
        vmambav0tiny = BuildModels.build_vmamba(with_ckpt=True, remove_head=True, scale="tv0").cuda().eval()
        vmambav2l5tiny = BuildModels.build_vmamba(with_ckpt=True, remove_head=True, scale="tv1").cuda().eval()
        vmambav2tiny = BuildModels.build_vmamba(with_ckpt=True, remove_head=True, scale="tv2").cuda().eval()
        convnexttiny = BuildModels.build_convnext(with_ckpt=True, remove_head=True, scale="tiny").cuda().eval()
        swintiny = BuildModels.build_swin_mmpretrain(with_ckpt=True, remove_head=True, scale="tiny", size=img_size).cuda().eval()
        hivittiny = BuildModels.build_hivit_mmpretrain(with_ckpt=True, remove_head=True, scale="tiny", size=img_size).cuda().eval()
        interntiny = BuildModels.build_intern(with_ckpt=True, remove_head=True, scale="tiny").cuda().eval()
        xcittiny = BuildModels.build_xcit(with_ckpt=True, remove_head=True, scale="tiny", size=img_size).cuda().eval()
        deitbase = BuildModels.build_deit_mmpretrain(with_ckpt=True, remove_head=True, scale="base", size=img_size).cuda().eval()
        # breakpoint()

    if False:
        resnet50 = None
        if True:
            print("resnet ================================", flush=True)
            model = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=False, with_norm=True,)
            model = model() 
            print(model.head.fc, flush=True)
            model.head.fc = nn.Identity() # 2048->1000
            model.cuda().eval()
            resnet50 = model
        
        deitsmall = None
        if True:
            print("deit small ================================", flush=True)
            model = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=False, with_norm=True,)
            model = model()
            print(model.head.layers.head, flush=True)
            model.head.layers.head = nn.Identity() # 384->1000
            model.cuda().eval()
            deitsmall = model

        vmambav0tiny = None
        if True:
            print("vmambav0 ================================", flush=True)
            _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
            model = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
            tckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
            model = model()
            model.load_state_dict(torch.load(open(tckpt, "rb"), map_location=torch.device("cpu"))["model"])
            print(model.classifier.head, flush=True)
            model.classifier.head = nn.Identity() # 768->1000
            model.cuda().eval()
            vmambav0tiny = model

        vmambav2l5tiny = None
        if True:
            print("vmambav2l5 ================================", flush=True)
            model = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
            tckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230/vssm1_tiny_0230_ckpt_epoch_262.pth"
            model = model()
            model.load_state_dict(torch.load(open(tckpt, "rb"), map_location=torch.device("cpu"))["model"])
            print(model.classifier.head, flush=True)
            model.classifier.head = nn.Identity() # 768->1000
            model.cuda().eval()
            vmambav2l5tiny = model

        vmambav2tiny = None
        if True:
            print("vmambav2 ================================", flush=True)
            import triton, mamba_ssm, selective_scan_cuda_oflex
            _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
            model = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
            tckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230s/vssm1_tiny_0230s_ckpt_epoch_264.pth"
            model = model()
            model.load_state_dict(torch.load(open(tckpt, "rb"), map_location=torch.device("cpu"))["model"])
            print(model.classifier.head, flush=True)
            model.classifier.head = nn.Identity() # 768->1000
            model.cuda().eval()
            vmambav2tiny = model

        convnexttiny = None
        if True:
            print("convnext ================================", flush=True)
            _model = import_abspy("convnext", f"{HOME}/OTHERS/ConvNeXt/models")
            model = _model.convnext_tiny()
            ckpt ="/home/LiuYue/Workspace/PylanceAware/ckpts/others/convnext_tiny_1k_224_ema.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
            print(model.head, flush=True)
            model.head = nn.Identity() # 768
            model.cuda().eval()
            convnexttiny = model

        swintiny = None
        if True:
            print("swin ================================", flush=True)
            from mmengine.runner import CheckpointLoader
            from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
            model = dict(
                type='ImageClassifier',
                backbone=dict(
                    type='SwinTransformer', arch='tiny', img_size=224, drop_path_rate=0.2),
                neck=dict(type='GlobalAveragePooling'),
                head=dict(
                    type='LinearClsHead',
                    num_classes=1000,
                    in_channels=768,
                    init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
                    loss=dict(
                        type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
                    cal_acc=False),
                init_cfg=[
                    dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                    dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
                ],
                train_cfg=dict(augments=[
                    dict(type='Mixup', alpha=0.8),
                    dict(type='CutMix', alpha=1.0)
                ]),
            )
            ckpt="https://download.openmmlab.com/mmclassification/v0/swin-transformer/swin_tiny_224_b16x64_300e_imagenet_20210616_090925-66df6be6.pth"
            model["backbone"].update({"window_size": int(img_size // 32)})
            model = build_classifier(model)
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
            print(model.head.fc, flush=True) # 768
            model.head.fc = nn.Identity()
            model.cuda().eval()
            swintiny = model

        hivittiny = None
        if True:
            from mmpretrain.models.builder import MODELS
            from mmengine.runner import CheckpointLoader
            from mmpretrain.models import build_classifier, ImageClassifier, HiViT, VisionTransformer, SwinTransformer
            from mmpretrain.models.backbones.vision_transformer import resize_pos_embed, to_2tuple, np
            
            @MODELS.register_module()
            class HiViTx(HiViT):
                def __init__(self, *args,**kwargs):
                    super().__init__(*args,**kwargs)
                    self.num_extra_tokens = 0
                    self.interpolate_mode = "bicubic"
                    self.patch_embed.init_out_size = self.patch_embed.patches_resolution
                    self._register_load_state_dict_pre_hook(self._prepare_abs_pos_embed)
                    self._register_load_state_dict_pre_hook(
                        self._prepare_relative_position_bias_table)

                # copied from SwinTransformer, change absolute_pos_embed to pos_embed
                def _prepare_abs_pos_embed(self, state_dict, prefix, *args, **kwargs):
                    name = prefix + 'pos_embed'
                    if name not in state_dict.keys():
                        return

                    ckpt_pos_embed_shape = state_dict[name].shape
                    if self.pos_embed.shape != ckpt_pos_embed_shape:
                        from mmengine.logging import MMLogger
                        logger = MMLogger.get_current_instance()
                        logger.info(
                            'Resize the pos_embed shape from '
                            f'{ckpt_pos_embed_shape} to {self.pos_embed.shape}.')

                        ckpt_pos_embed_shape = to_2tuple(
                            int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
                        pos_embed_shape = self.patch_embed.init_out_size

                        state_dict[name] = resize_pos_embed(state_dict[name],
                                                            ckpt_pos_embed_shape,
                                                            pos_embed_shape,
                                                            self.interpolate_mode,
                                                            self.num_extra_tokens)

                def _prepare_relative_position_bias_table(self, state_dict, *args, **kwargs):
                    del state_dict['backbone.relative_position_index']
                    return SwinTransformer._prepare_relative_position_bias_table(self, state_dict, *args, **kwargs)

            print("hivit ================================", flush=True)
            model = dict(
                backbone=dict(
                    ape=True,
                    arch='tiny',
                    drop_path_rate=0.05,
                    img_size=224,
                    rpe=True,
                    type='HiViTx'),
                head=dict(
                    cal_acc=False,
                    in_channels=384,
                    init_cfg=None,
                    loss=dict(
                        label_smooth_val=0.1, mode='original', type='LabelSmoothLoss'),
                    num_classes=1000,
                    type='LinearClsHead'),
                init_cfg=[
                    dict(bias=0.0, layer='Linear', std=0.02, type='TruncNormal'),
                    dict(bias=0.0, layer='LayerNorm', type='Constant', val=1.0),
                ],
                neck=dict(type='GlobalAveragePooling'),
                train_cfg=dict(augments=[
                    dict(alpha=0.8, type='Mixup'),
                    dict(alpha=1.0, type='CutMix'),
                ]),
                type='ImageClassifier')
            ckpt="/home/LiuYue/Workspace/PylanceAware/ckpts/others/hivit-tiny-p16_8xb128_in1k/epoch_295.pth"
            model["backbone"].update({"img_size": img_size})
            model = build_classifier(model)
            model.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
            print(model.head.fc, flush=True) # 768
            model.head.fc = nn.Identity()
            model.cuda().eval()
            hivittiny = model

        interntiny = None
        if True:
            print("intern ================================", flush=True)
            specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
            sys.path.insert(0, specpath)
            import DCNv3
            _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
            model = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
            model = model()
            ckpt ="/home/LiuYue/Workspace/PylanceAware/ckpts/others/internimage_t_1k_224.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
            sys.path = sys.path[1:]
            print(model.head, flush=True) # 768
            model.head = nn.Identity()
            model.cuda().eval()
            interntiny = model

        xcittiny = None
        if True:
            print("xcit ================================", flush=True)
            _model = import_abspy("xcit", f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/")
            model = _model.xcit_small_12_p16()
            ckpt =f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/others/xcit_small_12_p16_224.pth"
            model.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"])
            print(model.head, flush=True) # 768
            model.head = nn.Identity()
            model.cuda().eval()
            xcittiny = model

    if True:
        if step > 0:
            starts = list(range(start, end, step))
            ends = [s + step for s in starts]
            assert ends[-1] >= end
            ends[-1] = end
            print(f"multiple ranges: {starts} {ends} ==============", flush=True)
        else:
            starts, ends = [start], [end]

        for s, e in zip(starts, ends):
            extract_feature(
                backbones=dict(
                    # vmambav2tiny = vmambav2tiny,
                    # convnexttiny = convnexttiny,
                    # swintiny = swintiny,
                    # interntiny = interntiny,
                    # vmambav0tiny = vmambav0tiny,
                    # vmambav2l5tiny = vmambav2l5tiny,
                    # deitsmall = deitsmall,
                    # hivittiny = hivittiny,
                    # resnet50 = resnet50,
                    # xcittiny = xcittiny,
                    deitbase = deitbase,
                ), 
                dims=dict(
                    # vmambav2tiny = 768,
                    # convnexttiny = 768,
                    # swintiny = 768,
                    # interntiny = 768,
                    # vmambav0tiny = 768,
                    # vmambav2l5tiny = 768,
                    # deitsmall = 384,
                    # hivittiny = 384,
                    # resnet50 = 2048,
                    # xcittiny = 384,
                    deitbase = 768,
                ),
                batch_size=batch_size,
                img_size=img_size,
                data_path=data_path,
                ranges=(s, e),
                train=train,
                aug=aug,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, default="ImageNet_ILSVRC2012", help='path to dataset')
    parser.add_argument('--mode', type=str, default="", help='model name')
    parser.add_argument('--func', type=str, default="", help='function')
    parser.add_argument('--start', type=int, default=0, help='start range')
    parser.add_argument('--end', type=int, default=200, help='end range')
    parser.add_argument('--step', type=int, default=-1, help='step range')
    parser.add_argument('--size', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--val', action="store_true", help='...')
    parser.add_argument('--aug', action="store_true", help='...')
    args = parser.parse_args()
    print(args, flush=True)
    # breakpoint()
    
    modes = ["vssma6", "vssmaav1", "convnext", "resnet", "deit", "swin", "swinscale", "hivit", "intern", "xcit"]
    if args.mode != "":
        modes = [args.mode]
    
    if args.func == "scale":
       testscale(modes)
    elif args.func == "perf":
        testperf(modes)
    elif args.func == "feats":
        _extract_feature(args.data_path, args.start, args.end, args.step, args.size, args.batch_size, (not args.val), args.aug)
    else:
        raise NotImplementedError


def run_code_dist_one(func):
    if torch.cuda.device_count() > 1:
        print("WARNING!!!  acc score would be inaccurate if num_procs > 1, as sampler always pads the dataset")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # print(torch.cuda.device_count())
        exit()
        dist.init_process_group(backend='nccl', init_method='env://', world_size=-1, rank=-1)
    else:
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = "61234"
        while True:
            try:
                dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
                break
            except Exception as e:
                print(e, flush=True)
                os.environ['MASTER_PORT'] = f"{int(os.environ['MASTER_PORT']) - 1}"

    torch.cuda.set_device(dist.get_rank())
    dist.barrier()
    func()


if __name__ == "__main__":
    run_code_dist_one(main)


# CUDA_VISIBLE_DEVICES=0 python /home/LiuYue/Workspace/PylanceAware/VMamba/analyze/eval.py --func feats --start 0 --end 0 --size 224 --batch_size 128 \                               
