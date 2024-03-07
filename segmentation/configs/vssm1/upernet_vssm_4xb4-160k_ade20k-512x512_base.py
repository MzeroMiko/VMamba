_base_ = [
    '../swin/swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
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
    ),)
# train_dataloader = dict(batch_size=4) # as gpus=4

