function get_env() {
    pack_url=/home/LiuYue/Workspace3/Visualize/package
    conda_url="~/miniconda3"
    conda_env="mamba"
    nvcc -V
    bash ${pack_url}/Miniconda3-py310* -bfup ${conda_url}
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

# FAIED with mmcv-full, wasted
function get_env_mmdet1() {
    apex=../package/apex-master
    mmcv=../package/mmcv-1.3.18
    mmdet=../package/Swin-Transformer-Object-Detection-master
    mmseg=../package/Swin-Transformer-Semantic-Segmentation-main
    cd ${apex}; pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    cd ${mmcv}; MMCV_WITH_OPS=1 pip install .
    cd ${mmdet}; pip install . # failed with mmpycocotools! # modify requirements from mmpycocotools into pyccotools
    cd ${mmseg}; pip install .
}

# use this
function get_env_mmdet2() {
    pip install mmengine==0.10.1
    pip install mmcv==2.1.0
    pip install opencv-python-headless ftfy
    pip install mmdet==3.3.0
    pip install mmsegmentation==1.2.2
    pip install mmpretrain==1.2.0
}

# test ok
function dataset_to_memory() {
    # README: copy data into memory
    tar --use-compress-program=pigz -cvf ImageNet_ILSVRC2012.tar.pz ImageNet_ILSVRC2012/
    sudo mount -t tmpfs -o size=150G tmpfs .media/memfs/
    tar --use-compress-program=pigz -xvf ImageNet_ILSVRC2012.tar.pz -C /media/memfs/ # 5min
}

# test ok
function classification_tiny() {
    export CODE=classification PYTHON=python log_url="logs/classification"
    export nnodes=1 nrank=0 nprocs=8 mport=29501 maddr="127.0.0.1"
    export pycmds="main.py --cfg configs/vssm/vssm_tiny_224.yaml --batch-size 64 --data-path /dataset/ImageNet2012 --output /tmp"
    cd ${CODE}; ${PYTHON} -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} --log_dir ${log_url} ${pycmds}
}

# test ok
function classification_small() {
    export CODE=classification PYTHON=python log_url="logs/classification"
    export nnodes=1 nrank=0 nprocs=8 mport=29501 maddr="127.0.0.1"
    export pycmds="main.py --cfg configs/vssm/vssm_small_224.yaml --batch-size 64 --data-path /dataset/ImageNet2012 --output /tmp"
    cd ${CODE}; ${PYTHON} -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} --log_dir ${log_url} ${pycmds}
}

# test ok
function classification_base() {
    export CODE=classification PYTHON=python log_url="logs/classification"
    export nnodes=1 nrank=0 nprocs=8 mport=29501 maddr="127.0.0.1"
    export pycmds="main.py --cfg configs/vssm/vssm_base_224.yaml --batch-size 64 --data-path /dataset/ImageNet2012 --output /tmp"
    cd ${CODE}; ${PYTHON} -m torch.distributed.launch --nnodes ${nnodes} --node_rank ${nrank} --nproc_per_node ${nprocs} --master_addr ${maddr} --master_port ${mport} --log_dir ${log_url} ${pycmds}
}

# test ok
function detection() {
    mkdir -p detection/data
    ln -s /media/Disk1/Dataset/MSCOCO2017 detection/data/coco

    # test ok
    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_swin_fpn_coco_tiny.py
    export CKPT=../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok
    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_swin_fpn_coco_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ckpt_epoch_238.pth # TODO: use ema_ckpt_epoch_238.pth !!!
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT
    
    # test ok
    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_swin_fpn_coco_base.py
    export CKPT=../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT

    # test ok
    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_swin_fpn_coco_tiny_ms_3x.py
    export CKPT=../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok
    # ======================================================
    export CODE=detection PYTHON=python
    export CONFIG=configs/vssm/mask_rcnn_swin_fpn_coco_small_ms_3x.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ckpt_epoch_238.pth # TODO: use ema_ckpt_epoch_238.pth !!!
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

}

# test ok
function segmentation() {
    mkdir -p segmentation/data/ade
    ln -s /media/Disk1/Dataset/ADEChallengeData2016 segmentation/data/ade

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
    export CKPT=../../ckpts/classification/vssm/vssmtiny/ckpt_epoch_292.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py
    export CKPT=../../ckpts/classification/vssm/vssmbase/ckpt_epoch_260.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-896x896_small.py
    export CKPT=../../ckpts/classification/vssm/vssmsmall/ema_ckpt_epoch_238.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch --cfg-options model.backbone.pretrained=$CKPT 

    # test ok swin
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_swin_4xb4-160k_ade20k-640x640_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

    # test ok swin
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_swin_4xb4-160k_ade20k-896x896_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

    # test ok convnext
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_convnext_4xb4-160k_ade20k-640x640_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch

    # test ok convnext
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_convnext_4xb4-160k_ade20k-896x896_small.py
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/train.py $CONFIG --launcher pytorch
}

# test ok
function segmentation_test_tta() {
    mkdir -p segmentation/data/ade
    ln -s /media/Disk1/Dataset/ADEChallengeData2016 segmentation/data/ade

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_tiny/iter_144000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_small/iter_160000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-512x512_base/iter_128000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # test ok
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_vssm_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_vssm_4xb4-160k_ade20k-640x640_small/iter_112000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta --cfg-options model.backbone.pretrained=None

    # test ok swin
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_swin_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_swin_4xb4-160k_ade20k-640x640_small/iter_160000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta

    # test ok convnext
    # ======================================================
    export CODE=segmentation PYTHON=python
    export CONFIG=configs/vssm/upernet_convnext_4xb4-160k_ade20k-640x640_small.py
    export CKPT=../../ckpts/segmentation/work_dirs/upernet_convnext_4xb4-160k_ade20k-640x640_small/iter_160000.pth
    cd $CODE; export GPUS=8 NNODES=${NNODES:-1} NODE_RANK=${NODE_RANK:-0} PORT=${PORT:-29500} MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; PYTHONPATH="$PWD":$PYTHONPATH $PYTHON -m torch.distributed.launch --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT ./tools/test.py $CONFIG $CKPT --launcher pytorch --tta
}

function analyze() {
    # test ok =====================================================
    CUDA_VISIBLE_DEVICES=0 python analyze/get_erf.py > analyze/show/erf/get_erf.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python analyze/get_flops.py > analyze/show/flops/flops.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python analyze/get_loss.py
    # test ok =====================================================
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
    # test ok ============================================
    python analyze/scaleup_show.py
}




