function get_env() {
    conda_env="vmamba"
    nvcc -V
    conda create -n ${conda_env} --clone base
    conda init bash && source ~/.bashrc && conda activate vmamba
    python -VV
    pip -V
    pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
    # We use py110 cu117 torch113
    pip install packaging
    pip install timm==0.4.12
    pip install pytest chardet yacs termcolor
    pip install submitit tensorboardX
    pip install triton==2.0.0
    pip install fvcore
    pip install seaborn

    cd selective_scan && pip install . && pytest
    # you can also install packages below as an alternative...
    # pip install triton==2.0.0
    # pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
    # pip install mamba_ssm==1.0.1  # mamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
}

function get_env_mmdet() {
    pip install mmengine==0.10.1
    pip install mmcv==2.1.0
    pip install opencv-python-headless ftfy
    pip install mmdet==3.3.0
    pip install mmsegmentation==1.2.2
    pip install mmpretrain==1.2.0
}

function dataset_to_memory() {
    # README: copy data into memory
    tar --use-compress-program=pigz -cvf ImageNet_ILSVRC2012.tar.pz ImageNet_ILSVRC2012/
    sudo mount -t tmpfs -o size=150G tmpfs .media/memfs/
    tar --use-compress-program=pigz -xvf ImageNet_ILSVRC2012.tar.pz -C /media/memfs/ # 5min
}

function classification() {
    # ======================================================
    export CODE=classification PYTHON=python
    export nnodes=1 nrank=0 nprocs=8 mport=29501 maddr="127.0.0.1"
    export pycmds="main.py --cfg configs/vssm/vssm_tiny_224.yaml --batch-size 64 --data-path /dataset/ImageNet2012 --output /tmp"
    cd ${CODE}; ${PYTHON} -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} ${pycmds}

    # ======================================================
    export CODE=classification PYTHON=python
    export nnodes=1 nrank=0 nprocs=8 mport=29501 maddr="127.0.0.1"
    export pycmds="main.py --cfg configs/vssm/vssm_small_224.yaml --batch-size 64 --data-path /dataset/ImageNet2012 --output /tmp"
    cd ${CODE}; ${PYTHON} -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} ${pycmds}

    # ======================================================
    export CODE=classification PYTHON=python
    export nnodes=1 nrank=0 nprocs=8 mport=29501 maddr="127.0.0.1"
    export pycmds="main.py --cfg configs/vssm/vssm_base_224.yaml --batch-size 64 --data-path /dataset/ImageNet2012 --output /tmp"
    cd ${CODE}; ${PYTHON} -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} ${pycmds}
}

function detection() {
    mkdir -p detection/data
    ln -s /media/Disk1/Dataset/MSCOCO2017 detection/data/coco

    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py
    export CKPT=../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_vssm_fpn_coco_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ckpt_epoch_238.pth # TODO: use ema_ckpt_epoch_238.pth !!!
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT
    
    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_vssm_fpn_coco_base.py
    export CKPT=../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT

    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py
    export CKPT=../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_vssm_fpn_coco_small_ms_3x.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ckpt_epoch_238.pth # TODO: use ema_ckpt_epoch_238.pth !!!
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

}

function segmentation() {
    mkdir -p segmentation/data/ade
    ln -s /media/Disk1/Dataset/ADEChallengeData2016 segmentation/data/ade

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
    export CKPT=../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py
    export CKPT=../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-896x896_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_swin_4xb4-160k_ade20k-640x640_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_swin_4xb4-160k_ade20k-896x896_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_convnext_4xb4-160k_ade20k-640x640_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_convnext_4xb4-160k_ade20k-896x896_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch
}

function segmentation_test_tta() {
    mkdir -p segmentation/data/ade
    ln -s /media/Disk1/Dataset/ADEChallengeData2016 segmentation/data/ade

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_tiny/iter_144000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_small/iter_160000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_base/iter_128000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-640x640_small/iter_112000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_swin_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_swin_4xb4-160k_ade20k-640x640_small/iter_160000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta

    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_convnext_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_convnext_4xb4-160k_ade20k-640x640_small/iter_160000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta
}

function analyze() {
    CUDA_VISIBLE_DEVICES=0 python analyze/get_erf.py > analyze/show/erf/get_erf.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python analyze/get_flops.py > analyze/show/flops/flops.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python analyze/get_loss.py
    
    # =====================================================
    export ACTION=flops SCALENET=vssm; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/flops.log 2>&1
    export ACTION=tiny SCALENET=vssm; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/vssmtiny_scale.log 2>&1
    export ACTION=tiny SCALENET=swin; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/swintiny_scale.log 2>&1
    export ACTION=tiny SCALENET=convnext; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/convnexttiny_scale.log 2>&1
    export ACTION=tiny SCALENET=deit; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/deitsmall_scale.log 2>&1
    export ACTION=tiny SCALENET=resnet; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/resnet50_scale.log 2>&1

    export ACTION=small SCALENET=vssm; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/vssmsmall_scale.log 2>&1
    export ACTION=small SCALENET=swin; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/swinsmall_scale.log 2>&1
    export ACTION=small SCALENET=convnext; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/convnextsmall_scale.log 2>&1
    export ACTION=small SCALENET=resnet; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/resnet101_scale.log 2>&1
    
    export ACTION=base SCALENET=vssm; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/vssmbase_scale.log 2>&1
    export ACTION=base SCALENET=swin; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/swinbase_scale.log 2>&1
    export ACTION=base SCALENET=convnext; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/convnextbase_scale.log 2>&1
    export ACTION=base SCALENET=deit; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/deitbase_scale.log 2>&1
    export ACTION=base SCALENET=replknet; CUDA_VISIBLE_DEVICES=0 python analyze/get_scaleup.py >> analyze/show/scaleup.log/replknet31b_scale.log 2>&1
    
    # ============================================
    python analyze/scaleup_show.py
}



