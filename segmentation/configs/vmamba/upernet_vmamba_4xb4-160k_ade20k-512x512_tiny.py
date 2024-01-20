_base_ = [
    '../swin/swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MMSEG_VSSM',
        depths=(2, 2, 9, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/vssmtiny/ckpt_epoch_292.pth",
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4

