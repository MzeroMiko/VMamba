# this is only a script !
if __name__ == "__main__":
    import os
    from utils import visualize, EffectiveReceiptiveField, BuildModels
    simpnorm = EffectiveReceiptiveField.simpnorm

    HOME = os.environ["HOME"].rstrip("/")

    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./show").rstrip("/")
    data_path = "/media/Disk1/Dataset/ImageNet_ILSVRC2012"
    
    # main results
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_resnet_mmpretrain(only_backbone=True, scale="r50"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_convnext(only_backbone=True, scale="tiny"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_swin_mmpretrain(only_backbone=True, scale="tiny"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_deit_mmpretrain(only_backbone=True, scale="small"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_hivit_mmpretrain(only_backbone=True, scale="tiny", size=1024), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_resnet_mmpretrain(with_ckpt=True, only_backbone=True, scale="r50"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_convnext(with_ckpt=True, only_backbone=True, scale="tiny"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_swin_mmpretrain(with_ckpt=True, only_backbone=True, scale="tiny"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_deit_mmpretrain(with_ckpt=True, only_backbone=True, scale="small"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_hivit_mmpretrain(with_ckpt=True, only_backbone=True, scale="tiny", size=1024), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_main.jpg", rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')
    
    # erf for tv2, initv1, initv2; 
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_init="v1"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_init="v2"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_init="v1"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/classification/vssms/vssm1_tiny_0230s_init1/ckpt_epoch_261.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_init="v2"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/classification/vssms/vssm1_tiny_0230s_init2/ckpt_epoch_264.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_initv0v1v2.jpg", rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for tv2, unidi, bidi, cas2d; ndw, unidindw, bidindw, cas2dndw; 
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v052d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v052dc_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=True, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=False, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=False, forward_type="v052d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=False, forward_type="v052dc_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab1d/ckpt_epoch_272.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v052d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2d/ckpt_epoch_269.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v052dc_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2dc/ckpt_epoch_250.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=True, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230s_ndw/ckpt_epoch_237.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=False, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab1d_ndw/ckpt_epoch_272.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=False, forward_type="v052d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2d_ndw/ckpt_epoch_268.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=False, forward_type="v052dc_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/ckpts/private/classification/vssms/vssm1_tiny_0230ab2dc_ndw/ckpt_epoch_263.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_scanmethod.jpg", rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for training stage for tv2
    if False:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_2.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_5.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_8.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_11.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="flex", 
                cfg=dict(dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d"),
                ckpt=f"{HOME}/Workspace/PylanceAware/ckpts/private/vssm1_tiny_0230s/20240426174619/ckpt_epoch_29.pth",
                key="model_ema",
            ), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before, savefig=f"{showpath}/erf_trainprocess.jpg", rows=1, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for vim
    if True:
        from analyze_for_vim import ExtraDev
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(ExtraDev.build_vim_for_throughput(only_backbone=True, size=1024), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(ExtraDev.build_vim_for_throughput(with_ckpt=True, only_backbone=True, size=1024), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_vimmethods.jpg", rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')

    # erf for s4nd
    if True:
        results_before = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_s4nd(only_backbone=True, scale="ctiny"), size=1024, data_path=data_path, norms=simpnorm), ""),
        ]
        results_after = [
            (EffectiveReceiptiveField.get_input_grad_avg(BuildModels.build_vmamba(with_ckpt=True, only_backbone=True, scale="tv2"), size=1024, data_path=data_path, norms=simpnorm), ""),
            # (after_s4nd, ""),
        ]
        visualize.visualize_snsmaps(results_before + results_after, savefig=f"{showpath}/erf_s4ndmethods.jpg", rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn')




