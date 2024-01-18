function get_env() {
    pack_url=/home/LiuYue/Workspace3/Visualize/package
    conda_url=/home/LiuYue/miniconda3
    conda_env="mamba"
    nvcc -V
    bash ${pack_url}/Miniconda3-py310 -bfup ${conda_url}
    conda create -n ${conda_env} --clone base
    python -VV
    pip -V
    pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install packaging
    pip install timm==0.4.12
    pip install pytest chardet yacs termcolor
    pip install submitit tensorboardX
    pip install triton==2.0.0
    pip install ${pack_url}/causal_conv1d*
    pip install ${pack_url}/mamba_ssm*
}

# wasted
function get_env_mmdet1() {
    apex=/home/LiuYue/Workspace3/Visualize/package/apex-master
    mmcv=/home/LiuYue/Workspace3/Visualize/package/mmcv-1.3.18
    mmdet=/home/LiuYue/Workspace3/Visualize/package/Swin-Transformer-Object-Detection-master
    mmseg=/home/LiuYue/Workspace3/Visualize/package/Swin-Transformer-Semantic-Segmentation-main
    cd ${apex}; pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    cd ${mmcv}; MMCV_WITH_OPS=1 pip install .
    cd ${mmdet}; pip install . # failed with mmpycocotools! # modify requirements from mmpycocotools into pyccotools
    cd ${mmseg}; pip install .
}

function get_env_mmdet2() {
    pip install mmengine==0.10.1
    pip install mmcv==2.1.0
    pip install opencv-python-headless ftfy
    pip install mmdet==3.3.0
    pip install mmsegmentation==1.2.2
    pip install mmpretrain==1.2.0
}


# modify train.py: import model
# ln -s /media/Disk1/Dataset/MSCOCO2017 data/coco
# bash ./tools/dist_train.sh mask_rcnn_r50_fpn_coco.py 8

ln -sf /userhome/Data/COCO/ /userhome/liuyue/Visualize/analyze/detection/data/coco
ln -sf /userhome/Data/ADEChallengeData2016/ /userhome/liuyue/Visualize/analyze/segmentation/data/ade

cd /userhome/liuyue/Visualize/detection; export CONFIG=mask_rcnn_r50_fpn_coco.py GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=/userhome/liuyue/Visualize/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth

cd /userhome/liuyue/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512.py GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=/userhome/liuyue/Visualize/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth

# ============================
function analyze() {
    CUDA_VISIBLE_DEVICES=6 python analyze/get_erf.py > ./show/erf/get_erf.log 2>&1
    CUDA_VISIBLE_DEVICES=6 python analyze/get_flops.py > ./show/flops/flops.log 2>&1
    CUDA_VISIBLE_DEVICES=6 python analyze/get_loss.py
    CUDA_VISIBLE_DEVICES=6 GETSCALENET=vssm python analyze/get_scaleup.py >> ./show/scaleup.log/get_scaleup_vssm.log 2>&1
    CUDA_VISIBLE_DEVICES=5 GETSCALENET=swin python analyze/get_scaleup.py >> ./show/scaleup.log/get_scaleup_swin.log 2>&1
    CUDA_VISIBLE_DEVICES=4 GETSCALENET=convnext python analyze/get_scaleup.py >> ./show/scaleup.log/get_scaleup_sconvnext.log 2>&1
    CUDA_VISIBLE_DEVICES=6 GETSCALENET=replknet python analyze/get_scaleup.py >> ./show/scaleup.log/get_scaleup_replknet.log 2>&1
    CUDA_VISIBLE_DEVICES=4 GETSCALENET=deit python analyze/get_scaleup.py >> ./show/scaleup.log/get_scaleup_deit.log 2>&1
    CUDA_VISIBLE_DEVICES=6 GETSCALENET=resnet python analyze/get_scaleup.py >> ./show/scaleup.log/get_scaleup_resnet.log 2>&1
    CUDA_VISIBLE_DEVICES=6 python analyze/get_cam.py
    CUDA_VISIBLE_DEVICES=6 python analyze/get_throughput.py >> ./show/throughput.log/throughput.log 2>&1
}

# README: copy data into memory
# sudo mount -t tmpfs -o size=150G tmpfs .media/memfs/
# tar --use-compress-program=pigz -xvf ImageNet_ILSVRC2012.tar.pz -C /media/memfs/ # 5min
# tar --use-compress-program=pigz -cvf ImageNet_ILSVRC2012.tar.pz ImageNet_ILSVRC2012/

python -m torch.distributed.launch --nproc_per_node 7 --master_port 40001 main_ema.py --cfg configs/swin/swin_small_patch4_window7_224.yaml --batch-size 146  --data-path /media/memfs/ImageNet_ILSVRC2012/ --output ../../output --opts MODEL.TYPE vssmd MODEL.NAME vssmdsmall DATA.NUM_WORKERS 0 > trainsmall.log 2>&1 


cd /userhome/liuyue/newcodeversion/Visualize/classification; /userhome/liuyue/miniconda3/envs/mamba/bin/python -m torch.distributed.launch --nproc_per_node 16 --master_port 40001 main_ema.py --cfg configs/swin/swin_base_patch4_window7_224.yaml --batch-size 64  --data-path /dataset/ImageNet2012 --output ../../output --opts MODEL.TYPE vssmd MODEL.NAME vssmdbase

# =============================


# ======================================================

# det vssmtiny
cd /userhome/liuyue/newcodeversion/Visualize/detection; export CONFIG=mask_rcnn_swin_fpn_coco_tiny.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

# det vssmsmall
cd /userhome/liuyue/newcodeversion/Visualize/detection; export CONFIG=mask_rcnn_swin_fpn_coco_small.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

# det vssmbase
cd /userhome/liuyue/newcodeversion/Visualize/detection; export CONFIG=mask_rcnn_swin_fpn_coco_base.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

# seg vssmtiny
cd /userhome/liuyue/newcodeversion/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29501} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

# seg vssmsmall
cd /userhome/liuyue/newcodeversion/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512_small.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29501} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

# seg vssmbase
cd /userhome/liuyue/newcodeversion/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512_base.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29501} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

# seg vssmtiny test tta
cd /userhome/liuyue/newcodeversion/Visualize/segmentation; export CONFIG=upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py CKPT=/home/LiuYue/Workspace3/Visualize/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512/iter_160000.pth GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29501} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta

# ===============================
cd /userhome/liuyue/newcodeversion/Visualize/segmentation; export CONFIG=upernet_swin_4xb4-160k_ade20k-640x640_small.py GPUS=8 PYTHON=/userhome/liuyue/miniconda3/envs/mamba/bin/python NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29501} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch
