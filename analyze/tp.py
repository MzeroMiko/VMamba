import time
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

HOME = os.environ["HOME"].rstrip("/")
basicpath = os.path.abspath("../VMamba/analyze").rstrip("/")

# this mode will greatly influence the speed!
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


# default no amp in testing tp
@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        torch.cuda.reset_peak_memory_stats()
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return


@torch.no_grad()
def throughputamp(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            with torch.cuda.amp.autocast():
                model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        torch.cuda.reset_peak_memory_stats()
        tic1 = time.time()
        for i in range(30):
            with torch.cuda.amp.autocast():
                model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return


def testfwdbwd(data_loader, model, logger, amp=True):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()

    for idx, (images, targets) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(images)
                loss = criterion(out, targets)
                loss.backward()
        torch.cuda.synchronize()
        logger.info(f"testfwdbwd averaged with 30 times")
        torch.cuda.reset_peak_memory_stats()
        tic1 = time.time()
        for i in range(30):
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(images)
                loss = criterion(out, targets)
                loss.backward()
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} testfwdbwd {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return
    

def testall(model, dataloader, data_path, img_size=224, _batch_size=128, with_flops=False, inference_only=False):
    torch.cuda.empty_cache()
    model.cuda().eval()
    if with_flops:
        from flops import fvcore_flop_count
        fvcore_flop_count(model, input_shape=(3, img_size, img_size), show_arch=False)
    print(parameter_count(model)[""], flush=True)
    throughput(data_loader=dataloader, model=model, logger=logging)
    throughputamp(data_loader=dataloader, model=model, logger=logging) 
    if inference_only:
        return
    PASS = False
    batch_size = _batch_size
    while (not PASS) and (batch_size > 0):
        try:
            _dataloader = get_dataloader(
                batch_size=batch_size, 
                root=os.path.join(os.path.abspath(data_path), "val"),
                img_size=img_size,
            )
            testfwdbwd(data_loader=_dataloader, model=model, logger=logging)
            testfwdbwd(data_loader=_dataloader, model=model, logger=logging, amp=False)
            PASS = True
        except:
            batch_size = batch_size // 2
            print(f"batch_size {batch_size}", flush=True)


def get_variable_name(variable, loc=locals()):
    for k,v in loc.items():
        if loc[k] is variable:
            return k


def main0():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--size', type=int, default=224, help='path to dataset')
    args = parser.parse_args()
    modes = ["convnexts4nd"]
    modes = ["vssm", "resnet", "deit", "vim", "swin", "convnext", "hivit", "intern"]
    modes = ["deit", "convnexts4nd"]
    modes = ["deit"]

    logging.basicConfig(level=logging.INFO)

    dataloader = get_dataloader(
        batch_size=args.batch_size, 
        root=os.path.join(os.path.abspath(args.data_path), "val"),
        img_size=args.size,
    )

    # convnext-s4nd: this needs timm=0.5.4; install extentions/kernel
    if "convnexts4nd" in modes:
        print("convnext-s4nd ================================", flush=True)
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./convnexts4nd")
        sys.path.insert(0, specpath)
        import timm; assert timm.__version__ == "0.5.4"
        import structured_kernels
        model = import_abspy("vit_all", f"{basicpath}/convnexts4nd")
        testall(model.vit_base_s4nd(), dataloader, args.data_path, args.size, args.batch_size)
        model = import_abspy("convnext_timm", f"{basicpath}/convnexts4nd")
        testall(model.convnext_tiny_s4nd(), dataloader, args.data_path, args.size, args.batch_size)
        sys.path = sys.path[1:]

    # vim: install mamba_ssm
    if "vim" in modes:
        print("vim ================================", flush=True)
        specpath = f"{HOME}/OTHERS/Vim/mamba-1p1p1"
        sys.path.insert(0, specpath)
        import mamba_ssm
        model = import_abspy("models_mamba", f"{HOME}/OTHERS/Vim/vim")
        testall(model.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2(), dataloader, args.data_path, args.size, args.batch_size)
        sys.path = sys.path[1:]

    # deit
    if "deit" in modes:
        _build = import_abspy("models", f"{basicpath}/../classification")
        build_mmpretrain_models = _build.build_mmpretrain_models
        print("deit ================================", flush=True)
        tiny = partial(build_mmpretrain_models, cfg="deit_small", ckpt=False, only_backbone=False, with_norm=True,)
        base = partial(build_mmpretrain_models, cfg="deit_base", ckpt=False, only_backbone=False, with_norm=True,)
        base384 = partial(build_mmpretrain_models, cfg="deit_base", ckpt=False, only_backbone=False, with_norm=True,)
        for config in [tiny, base, base384]:
            size = args.size if not config == base384 else 384
            testall(config(), dataloader, args.data_path, size, args.batch_size, with_flops=True)

    # swin: install kernels/window_process
    if "swin" in modes:
        print("swin ================================", flush=True)
        specpath = f"{HOME}/OTHERS/Swin-Transformer"
        sys.path.insert(0, specpath)
        import swin_window_process
        _model = import_abspy("swin_transformer", f"{HOME}/OTHERS/Swin-Transformer/models")
        # configs/swin/swin_tiny_patch4_window7_224.yaml
        tiny = partial(_model.SwinTransformer, embed_dim=96, depths=[2,2,6,2], num_heads=[ 3, 6, 12, 24 ], fused_window_process=True)
        # configs/swin/swin_small_patch4_window7_224.yaml
        small = partial(_model.SwinTransformer, embed_dim=96, depths=[2,2,18,2], num_heads=[ 3, 6, 12, 24 ], fused_window_process=True)
        # # configs/swin/swin_base_patch4_window7_224.yaml
        base = partial(_model.SwinTransformer, embed_dim=128, depths=[2,2,18,2], num_heads=[ 4, 8, 16, 32 ], fused_window_process=True)

        for config in [tiny, small, base]:
            testall(config(), dataloader, args.data_path, args.size, args.batch_size)
        sys.path = sys.path[1:]

    # convnext:
    if "convnext" in modes:
        print("convnext ================================", flush=True)
        sys.path.insert(0, "")
        _model = import_abspy("convnext", f"{HOME}/OTHERS/ConvNeXt/models")
        tiny = _model.convnext_tiny
        small = _model.convnext_small
        base = _model.convnext_base
        for config in [tiny, small, base]:
            testall(config(), dataloader, args.data_path, args.size, args.batch_size)
        sys.path = sys.path[1:]

    # hivit:
    if "hivit" in modes:
        print("hivit ================================", flush=True)
        sys.path.insert(0, "")
        _model = import_abspy("hivit", f"{HOME}/OTHERS/hivit/supervised/models/")
        tiny = partial(_model.HiViT, patch_size=16, inner_patches=4, embed_dim=384, depths=[1, 1, 10], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        small = partial(_model.HiViT, patch_size=16, inner_patches=4, embed_dim=384, depths=[2, 2, 20], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        base = partial(_model.HiViT, patch_size=16, inner_patches=4, embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        
        for config in [tiny, small, base]:
            testall(config(), dataloader, args.data_path, args.size, args.batch_size)
        sys.path = sys.path[1:]

    # internimage: install classification/ops_dcnv3
    if "intern" in modes:
        print("intern ================================", flush=True)
        specpath = f"{HOME}/OTHERS/InternImage/classification"
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        tiny = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        small = partial(_model.InternImage, core_op='DCNv3', channels=80, depths=[4, 4, 21, 4], groups=[5, 10, 20, 40], layer_scale=1e-5, offset_scale=1.0, mlp_ratio=4., post_norm=True)
        base = partial(_model.InternImage, core_op='DCNv3', channels=112, depths=[4, 4, 21, 4], groups=[7, 14, 28, 56], layer_scale=1e-5, offset_scale=1.0, mlp_ratio=4., post_norm=True)
        
        for config in [tiny, small, base]:
            testall(config(), dataloader, args.data_path, args.size, args.batch_size)
        sys.path = sys.path[1:]


def main1():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--size', type=int, default=224, help='path to dataset')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    # modes = ["vssma6", "vssmaav1", "swin", "swinscale", "convnext", "hivit", "intern","deit", "resnet"]
    modes = ["swin", "swinscale"]
    # modes = ["intern"]

    _build = import_abspy("models", f"{basicpath}/../classification")
    build_mmpretrain_models = _build.build_mmpretrain_models

    def test_size(config):
        for size in [224, 384, 512, 640, 768, 1024]:
            print(f"testing size {size}...")
            dataloader = get_dataloader(
                batch_size=args.batch_size, 
                root=os.path.join(os.path.abspath(args.data_path), "val"),
                img_size=size,
            )
            try:
                model = config(img_size=size)
            except Exception as e:
                print(e, flush=True)
                model = config()
            # in most cases, it works
            testall(model, dataloader, args.data_path, size, args.batch_size)

    # vssm ta6: install selective_scan
    if "vssma6" in modes:
        print("vssm ta6 ================================", flush=True)
        import triton, mamba_ssm, selective_scan_cuda_oflex
        _model = import_abspy("vmamba", f"{basicpath}/../classification/models")
        ta6 = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
        test_size(ta6)

    # vssm taav1: install selective_scan
    if "vssmaav1" in modes:
        print("vssm taav1 ================================", flush=True)
        taav1 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
        test_size(taav1)
    
    # resnet
    if "resnet" in modes:
        print("resnet ================================", flush=True)
        tiny = partial(build_mmpretrain_models, cfg="resnet50", ckpt=False, only_backbone=False, with_norm=True,)
        test_size(tiny)

    # deit
    if "deit" in modes:
        print("deit ================================", flush=True)
        tiny = partial(build_mmpretrain_models, cfg="deit_small", ckpt=False, only_backbone=False, with_norm=True,)
        test_size(tiny)

    # convnext
    if "convnext" in modes:
        print("convnext ================================", flush=True)
        _model = import_abspy("convnext", f"{HOME}/OTHERS/ConvNeXt/models")
        tiny = _model.convnext_tiny
        test_size(tiny)

    # hivit
    if "hivit" in modes:
        print("hivit ================================", flush=True)
        _model = import_abspy("hivit", f"{HOME}/OTHERS/hivit/supervised/models/")
        tiny = partial(_model.HiViT, patch_size=16, inner_patches=4, embed_dim=384, depths=[1, 1, 10], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        test_size(tiny)

    # swin
    if "swin" in modes:
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
        for size in [224, 384, 512, 640, 768, 1024]:
            model["backbone"].update({"window_size": 7, "img_size": size})
            tiny = build_classifier(model)
            print(f"testing size {size}...")
            dataloader = get_dataloader(
                batch_size=args.batch_size, 
                root=os.path.join(os.path.abspath(args.data_path), "val"),
                img_size=size,
            )
            testall(tiny, dataloader, args.data_path, size, args.batch_size)

    # swin
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
        for size in [224, 384, 512, 640, 768, 1024]:
            model["backbone"].update({"window_size": int(size // 32), "img_size": size})
            tiny = build_classifier(model)
            print(f"testing size {size}...")
            dataloader = get_dataloader(
                batch_size=args.batch_size, 
                root=os.path.join(os.path.abspath(args.data_path), "val"),
                img_size=size,
            )
            testall(tiny, dataloader, args.data_path, size, args.batch_size)

    # internimage: install classification/ops_dcnv3
    if "intern" in modes:
        print("intern ================================", flush=True)
        specpath = f"{HOME}/OTHERS/InternImage/classification"
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        tiny = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        test_size(tiny)
        sys.path = sys.path[1:]


if __name__ == "__main__":
    main0()


    
