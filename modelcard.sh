function dataset_to_memory() {
    # README: copy data into memory
    tar --use-compress-program=pigz -cvf ImageNet_ILSVRC2012.tar.pz ImageNet_ILSVRC2012/
    sudo mount -t tmpfs -o size=150G tmpfs .media/memfs/
    tar --use-compress-program=pigz -xvf ImageNet_ILSVRC2012.tar.pz -C /media/memfs/ # 5min
}

function classification() {
    # VMambav0-T ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav0_tiny_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out

    # VMambav0-S ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav0_small_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out

    # VMambav0-B ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav0_base_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out
    
    # VMamba-T[s2l5] ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav2_tiny_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out
       
    # VMamba-T ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav2v_tiny_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out

    # VMamba-S ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav2_small_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out
       
    # VMamba-S[s1l20] ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav2v_small_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out

    # VMamba-B ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav2_base_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out
       
    # VMamba-B[s1l20] ======================================================
    torchrun --nproc_per_node 8 --master_port 12340 main.py --cfg configs/vssm/vmambav2v_base_224.yaml --data-path /media/memfs/ImageNet_ILSVRC2012 --output ../../out
}

function detection() {
    mkdir -p detection/data
    ln -s /media/Disk1/Dataset/MSCOCO2017 detection/data/coco
    cd detection

    # VMambav0-T ======================================================
    export CKPT="publish/vssm0/classification/vssm0_tiny/vssm0_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/mask_rcnn_vssm_fpn_coco_tiny.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMambav0-S ======================================================
    export CKPT="publish/vssm0/classification/vssm0_small/vssm0_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/mask_rcnn_vssm_fpn_coco_small.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMambav0-B ======================================================
    export CKPT="publish/vssm0/classification/vssm0_base/vssm0_base_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/mask_rcnn_vssm_fpn_coco_base.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-T[s2l5] ======================================================
    export CKPT="publish/vssm1/classification/vssm1_tiny/vssm1_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny1.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-T ======================================================
    export CKPT="publish/vssm2/classification/vssm2_tiny/vssm2_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-S ======================================================
    export CKPT="publish/vssm1/classification/vssm1_small/vssm1_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_small.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-B ======================================================
    export CKPT="publish/vssm1/classification/vssm1_base/vssm1_base_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py 8 --cfg-options model.backbone.pretrained=$CKPT
}

function detection_ms3x() {
    mkdir -p detection/data
    ln -s /media/Disk1/Dataset/MSCOCO2017 detection/data/coco
    cd detection

    # VMambav0-T ======================================================
    export CKPT="publish/vssm0/classification/vssm0_tiny/vssm0_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMambav0-S ======================================================
    export CKPT="publish/vssm0/classification/vssm0_small/vssm0_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/mask_rcnn_vssm_fpn_coco_small_ms_3x.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-T[s2l5] ======================================================
    export CKPT="publish/vssm1/classification/vssm1_tiny/vssm1_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny1_ms_3x.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-T ======================================================
    export CKPT="publish/vssm2/classification/vssm2_tiny/vssm2_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_tiny_ms_3x.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-S ======================================================
    export CKPT="publish/vssm1/classification/vssm1_small/vssm1_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/mask_rcnn_vssm_fpn_coco_small_ms_3x.py 8 --cfg-options model.backbone.pretrained=$CKPT
}

function segmentation() {
    mkdir -p segmentation/data/ade
    ln -s /media/Disk1/Dataset/ADEChallengeData2016 segmentation/data/ade
    cd detection

    # VMambav0-T ======================================================
    export CKPT="publish/vssm0/classification/vssm0_tiny/vssm0_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMambav0-S ======================================================
    export CKPT="publish/vssm0/classification/vssm0_small/vssm0_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMambav0-B ======================================================
    export CKPT="publish/vssm0/classification/vssm0_base/vssm0_base_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-T[s2l5] ======================================================
    export CKPT="publish/vssm1/classification/vssm1_tiny/vssm1_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny1.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-T ======================================================
    export CKPT="publish/vssm2/classification/vssm2_tiny/vssm2_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-S ======================================================
    export CKPT="publish/vssm1/classification/vssm1_small/vssm1_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py 8 --cfg-options model.backbone.pretrained=$CKPT

    # VMamba-B ======================================================
    export CKPT="publish/vssm1/classification/vssm1_base/vssm1_base_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_train.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py 8 --cfg-options model.backbone.pretrained=$CKPT
}

function segmentation_test_tta() {
    mkdir -p segmentation/data/ade
    ln -s /media/Disk1/Dataset/ADEChallengeData2016 segmentation/data/ade
    cd segmentation

    # VMambav0-T ======================================================
    export CKPT="publish/vssm0/segmentation/vssm0_tiny/vssm0_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta

    # VMambav0-S ======================================================
    export CKPT="publish/vssm0/segmentation/vssm0_small/vssm0_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_small.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta

    # VMambav0-B ======================================================
    export CKPT="publish/vssm0/segmentation/vssm0_base/vssm0_base_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm/upernet_vssm_4xb4-160k_ade20k-512x512_base.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta

    # VMamba-T[s2l5] ======================================================
    export CKPT="publish/vssm1/segmentation/vssm1_tiny/vssm1_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny1.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta

    # VMamba-T ======================================================
    export CKPT="publish/vssm2/segmentation/vssm2_tiny/vssm2_tiny_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta

    # VMamba-S ======================================================
    export CKPT="publish/vssm1/segmentation/vssm1_small/vssm1_small_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_small.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta

    # VMamba-B ======================================================
    export CKPT="publish/vssm1/segmentation/vssm1_base/vssm1_base_ckpt_epoch.pth"
    PORT=12345 bash ./tools/dist_test.sh configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_base.py 8 --cfg-options model.backbone.pretrained=$CKPT --tta
}
