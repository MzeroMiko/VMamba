_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MMDET_VSSM',
        depths=(2, 2, 27, 2),
        dims=128,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/vssmbase/ckpt_epoch_260.pth",
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)

# too big
train_dataloader = dict(batch_size=1) # as gpus=16



