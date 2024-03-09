_base_ = [
    '../swin/mask-rcnn_swin-t-p4-w7_fpn_1x_coco.py'
]

model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained="/home/zjy/nodeHPC9/downstream/vssm1_base_0229/ckpt_epoch_237.pth",
        # copied from classification/configs/vssm/vssm_base_224.yaml
        dims=128,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v3noz",
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.6,
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)

# too big
# train_dataloader = dict(batch_size=1) # as gpus=16



