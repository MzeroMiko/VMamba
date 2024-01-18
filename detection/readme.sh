# modify train.py: import model

function set_env() {
    pip install mmengine==0.10.1
    pip install mmcv==2.1.0
    pip install mmdet==3.3.0   
    pip install opencv-python-headless ftfy
    pip install mmsegmentation==1.2.2
}


# ln -s /media/Disk1/Dataset/MSCOCO2017 data/coco
# bash ./tools/dist_train.sh mask_rcnn_r50_fpn_coco.py 8

ln -sf /userhome/Data/COCO/ /userhome/liuyue/Visualize/detection/data/coco
ln -sf /userhome/Data/ADEChallengeData2016/ /userhome/liuyue/Visualize/detection/data/coco
>> /userhome/liuyue/debug.log 2>&1

cd /userhome/liuyue/Visualize/detection; export CONFIG=mask_rcnn_r50_fpn_coco.py GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=/userhome/liuyue/Visualize/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth

cd /userhome/liuyue/Visualize/detection; export CONFIG=mask_rcnn_r50_fpn_coco.py GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=/userhome/liuyue/Visualize/ckpts/ckpt_vssm_base_224/ckpt_epoch_260.pth


cd /userhome/liuyue/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512.py GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=/userhome/liuyue/Visualize/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth

cd /userhome/liuyue/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512.py GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=/userhome/liuyue/Visualize/ckpts/ckpt_vssm_base_224/ckpt_epoch_260.pth



