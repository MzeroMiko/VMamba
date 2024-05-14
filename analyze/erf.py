import os
from utils import visualize, EffectiveReceiptiveField, BuildModels
simpnorm = EffectiveReceiptiveField.simpnorm


if __name__ == "__main__":
    showpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./show/erf.jpg")
    data_path = "/media/Disk1/Dataset/ImageNet_ILSVRC2012"
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

    visualize.visualize_snsmaps(
        results_before + results_after, savefig=showpath, rows=2, sticks=False, figsize=(10, 10.75), cmap='RdYlGn', 
    )



