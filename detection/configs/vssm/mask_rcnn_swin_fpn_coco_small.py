_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MMDET_VSSM',
        depths=(2, 2, 27, 2),
        dims=96,
        out_indices=(0, 1, 2, 3),
        pretrained="../../ckpts/vssmsmall/ckpt_epoch_292.pth",
    ),
)

# train_dataloader = dict(batch_size=2) # as gpus=8

