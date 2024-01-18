_base_ = [
    '../swin/swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]
model = dict(
    backbone=dict(
        type='MMSEG_VSSM',
        depths=(2, 2, 27, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/vssmsmall/ema_ckpt_epoch_238.pth",
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4

