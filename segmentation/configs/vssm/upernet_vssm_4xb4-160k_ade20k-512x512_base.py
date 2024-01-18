_base_ = [
    '../swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MMSEG_VSSM',
        depths=(2, 2, 27, 2),
        dims=128,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/vssmbase/ckpt_epoch_260.pth",
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4

