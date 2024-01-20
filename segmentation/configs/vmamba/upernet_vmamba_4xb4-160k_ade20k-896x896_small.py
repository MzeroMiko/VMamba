_base_ = [
    './upernet_swin_4xb4-160k_ade20k-896x896_small.py'
]
model = dict(
    backbone=dict(
        type='MMSEG_VSSM',
        depths=(2, 2, 27, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/vssmsmall/ckpt_epoch_238.pth",
    ),)
train_dataloader = dict(batch_size=4) # as gpus=4

