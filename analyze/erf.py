import os
import sys
from functools import partial
import torch
import torch.nn as nn


from utils import visualize, EffectiveReceiptiveField, import_abspy
visualize_attnmap = visualize.visualize_attnmap
visualize_attnmaps = visualize.visualize_attnmaps
simpnorm = EffectiveReceiptiveField.simpnorm
get_input_grad_avg = EffectiveReceiptiveField.get_input_grad_avg


def main():
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./show/erf.jpg")
    data_path = "/media/Disk1/Dataset/ImageNet_ILSVRC2012"
    results_before = []
    results_after = []

    # modes = ["resnet", "convnext", "intern", "swin", "hivit", "deit", "vssma6", "vssmaav1"]
    modes = ["resnet", "convnext", "swin", "deit", "hivit", "vssmaav1"]

    _build = import_abspy("models", f"{os.path.dirname(__file__)}/../classification")
    build_mmpretrain_models = _build.build_mmpretrain_models

    def vssm_backbone(model, permute=False):
        class Permute(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
            def forward(x):
                return x.permute(0, 3, 1, 2)
        model.classifier = Permute() if permute else nn.Identity()
        return model

    def intern_backbone(model):
        def forward(self, x):
            x = self.patch_embed(x)
            x = self.pos_drop(x)

            for level in self.levels:
                x = level(x)
            return x.permute(0, 3, 1, 2)
        
        model.forward = partial(forward, model)
        return model

    def mmpretrain_backbone(model, with_norm=False):
        from mmpretrain.models import build_classifier, ImageClassifier, ConvNeXt, VisionTransformer, SwinTransformer
        if isinstance(model.backbone, ConvNeXt):
            model.backbone.gap_before_final_norm = False
        if isinstance(model.backbone, VisionTransformer):
            model.backbone.out_type = 'featmap'

        def forward_backbone(self: ImageClassifier, x):
            x = self.backbone(x)[-1]
            return x
        if not with_norm:
            setattr(model, f"norm{model.backbone.out_indices[-1]}", lambda x: x)
        model.forward = partial(forward_backbone, model)
        return model

    if "resnet" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        model_before = partial(build_mmpretrain_models, cfg="resnet50", ckpt=False, only_backbone=True, with_norm=False)()
        model_after = partial(build_mmpretrain_models, cfg="resnet50", ckpt=True, only_backbone=True, with_norm=False)()
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "convnext" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        model_before = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=False, only_backbone=True, with_norm=False,)()
        model_after = partial(build_mmpretrain_models, cfg="convnext_tiny", ckpt=True, only_backbone=True, with_norm=False,)()
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "intern" in modes:
        HOME = os.environ["HOME"].rstrip("/")
        model_name = ""
        print("intern ================================", flush=True)
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
        sys.path.insert(0, specpath)
        import DCNv3
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        model = partial(_model.InternImage, core_op='DCNv3', channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        model_before = intern_backbone(model())
        model_after = intern_backbone(model())
        ckpt ="/home/LiuYue/Workspace/PylanceAware/ckpts/others/internimage_t_1k_224.pth"
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        sys.path = sys.path[1:]

    if "swin" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
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
        model["backbone"].update({"img_size": 1024})
        model_before = mmpretrain_backbone(build_classifier(model))
        model_after = mmpretrain_backbone(build_classifier(model))
        model_after.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
    
    if "deit" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        model_before = partial(build_mmpretrain_models, cfg="deit_small", ckpt=False, only_backbone=True, with_norm=False,)()
        model_after = partial(build_mmpretrain_models, cfg="deit_small", ckpt=True, only_backbone=True, with_norm=False,)()
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "hivit" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
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
        model["backbone"].update({"img_size": 1024})
        model_before = mmpretrain_backbone(build_classifier(model))
        model_after = mmpretrain_backbone(build_classifier(model))
        model_after.load_state_dict(CheckpointLoader.load_checkpoint(ckpt)['state_dict'], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "vssma6" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        ta6 = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d")
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm/classification/vssmtiny/vssmtiny_dp01_ckpt_epoch_292.pth"
        model_before = vssm_backbone(ta6().cuda().eval())
        model_after = vssm_backbone(ta6().cuda().eval())
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])

    if "vssmaav1" in modes:
        model_name = ""
        print(f"{model_name} ================================", flush=True)
        _model = import_abspy("vmamba", f"{os.path.dirname(__file__)}/../classification/models")
        taav1 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")
        ckpt = "/home/LiuYue/Workspace/PylanceAware/ckpts/publish/vssm1/classification/vssm1_tiny_0230/vssm1_tiny_0230_ckpt_epoch_262.pth"
        model_before = vssm_backbone(taav1().cuda().eval())
        model_after = vssm_backbone(taav1().cuda().eval())
        model_after.load_state_dict(torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))["model"], strict=False)
        results_before.extend([
            (get_input_grad_avg(model_before, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
        results_after.extend([
            (get_input_grad_avg(model_after, size=1024, data_path=data_path, norms=simpnorm), model_name)
        ])
    

    visualize.visualize_snsmaps(
        results_before + results_after, savefig=showpath, rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn', 
    )
        

if __name__ == "__main__":
    main()



