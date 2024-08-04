import torch
import torch.utils.data
import argparse
import os
import logging
from functools import partial

HOME = os.environ["HOME"].rstrip("/")
basicpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "../analyze")).rstrip("/")
print(basicpath)

# this mode will greatly influence the speed!
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


# utils = import_abspy(name="utils", path=f"{basicpath}")
# BuildModels = utils.BuildModels
# FLOPs = utils.FLOPs
# Throughput = utils.Throughput
# get_val_dataloader = utils.get_val_dataloader

from utils import BuildModels, FLOPs, Throughput, get_val_dataloader


def get_variable_name(variable, loc=locals()):
    for k,v in loc.items():
        if loc[k] is variable:
            return k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--size', type=int, default=224, help='path to dataset')
    # parser.add_argument('--mode', type=str, default="", help='model name')
    args = parser.parse_args()
        
    logging.basicConfig(level=logging.INFO)

    def test_scaleup(build=None, sizes=[224, 288, 256, 384, 512, 640, 768, 1024], batch_size=32, data_path=args.data_path):
        for size in sizes:
            Throughput.testall(build(size=size), None, data_path, size, batch_size)

    dataloader = get_val_dataloader(
        batch_size=args.batch_size, 
        root=os.path.join(os.path.abspath(args.data_path), "val"),
        img_size=args.size,
    )
    size = args.size

    if False:
        Throughput.testall(BuildModels.build_vmamba(scale="tv0"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_vmamba(scale="tv1"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_vmamba(scale="tv2"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_vmamba(scale="sv0"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_vmamba(scale="sv2"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_vmamba(scale="bv0"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_vmamba(scale="bv2"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_visionmamba(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_deit_mmpretrain(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_deit_mmpretrain(scale="base"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_swin(scale="tiny"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_swin(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_swin(scale="base"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_convnext(scale="tiny"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_convnext(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_convnext(scale="base"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_hivit(scale="tiny"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_hivit(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_hivit(scale="base"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_intern(scale="tiny"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_intern(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_intern(scale="base"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_xcit(scale="tiny"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_xcit(scale="small"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_xcit(scale="base"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_resnet_mmpretrain(scale="r50"), dataloader, args.data_path, size, args.batch_size)
        Throughput.testall(BuildModels.build_resnet_mmpretrain(scale="r101"), dataloader, args.data_path, size, args.batch_size)
        return

    if False:
        # T15: GFlops:  4.905609984 Params:  30249064 1666; 3057; 564;  12391; 450; 20857;
        Throughput.testall(BuildModels.build_vmamba(scale="tv2"), dataloader, args.data_path, size, args.batch_size)
        _model = import_abspy("vmamba", f"{basicpath}/../classification/models")
        # T15: GFlops:  4.018473215999999 Params:  22926376 1336; 3602; 405;  10314; 367; 17234;
        abt_tv1_161_2222_mlp = partial(_model.VSSM, dims=96, depths=[2,2,2,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1,  forward_type="v05", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # 
        Throughput.testall(abt_tv1_161_2222_mlp(), dataloader, args.data_path, size, args.batch_size)

    if False:
        from analyze_for_vim import ExtraDev
        for size in [224, 288, 256, 384, 512, 640, 768, 1024]:
            print(f"s4nd {size} ==========================")
            ExtraDev.flops_s4nd(size=size, scale="ctiny")
            Throughput.testall(BuildModels.build_s4nd(size=size), None, args.data_path, size, 32, with_flops=False)
        return

    if False:
        from analyze_for_vim import ExtraDev
        for size in [224, 288, 256, 384, 512, 640, 768, 1024]:
            print(f"vim {size} ==========================")
            ExtraDev.flops_vim(size=size)
            Throughput.testall(ExtraDev.build_vim_for_throughput(size=size), None, args.data_path, size, 32, with_flops=False)
        return

    if False:
        test_scaleup(partial(BuildModels.build_vmamba, scale="tv2"))
        return

    if True:
        # 3056 12385 # structure difference
        Throughput.testall(BuildModels.build_vmamba(scale="tv2"), dataloader, args.data_path, size, args.batch_size)
        # 1111 9103 # structure difference
        Throughput.testall(BuildModels.build_visionmamba(scale="small"), dataloader, args.data_path, size, args.batch_size)
        # tmem 3945 mem 18782
        Throughput.testall(BuildModels.build_s4nd(scale="ctiny"), dataloader, args.data_path, size, args.batch_size)
        # tmem 2463 mem 15328
        Throughput.testall(BuildModels.build_s4nd(scale="vitb"), dataloader, args.data_path, size, args.batch_size)
        return

    if True:
        test_scaleup(build=partial(BuildModels.build_vmamba, scale="tv0"))
        test_scaleup(build=partial(BuildModels.build_vmamba, scale="tv1"))
        test_scaleup(build=partial(BuildModels.build_vmamba, scale="tv2"))
        test_scaleup(build=partial(BuildModels.build_convnext, scale="tiny"))
        test_scaleup(build=partial(BuildModels.build_swin, scale="tiny"))
        test_scaleup(build=partial(BuildModels.build_hivit, scale="tiny"))
        test_scaleup(build=partial(BuildModels.build_intern, scale="tiny"))
        test_scaleup(build=partial(BuildModels.build_xcit, scale="tiny"))
        test_scaleup(build=partial(BuildModels.build_deit_mmpretrain, scale="small"))
        test_scaleup(build=partial(BuildModels.build_resnet_mmpretrain, scale="r50"))
        return

    if True:
        for size in [224, 288, 256, 384, 512, 640, 768]:
            _dataloader = get_val_dataloader(
                batch_size=args.batch_size, 
                root=os.path.join(os.path.abspath(args.data_path), "val"),
                img_size=size,
            )
            Throughput.testall(BuildModels.build_swin(scale="base", size=size), _dataloader, args.data_path, size, batch_size)
            Throughput.testall(BuildModels.build_vheat(scale="base", size=size), _dataloader, args.data_path, size, batch_size)
        return

    if True:
        _model = import_abspy("vmamba", f"{basicpath}/../classification/models")
        tv0 = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v0", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1") # GFlops:  5.62689504 Params:  22893448 404
        tv1 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") #GFlops:  4.8577420799999995 Params:  30705832 1269
        tv2 = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d")  # GFlops:  4.905609984 Params:  30249064  1603
        sv0 = partial(_model.VSSM, dims=96, depths=[2,2,27,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v0", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1") # GFlops:  11.231522784000001 Params:  44417416 219
        sv1 = partial(_model.VSSM, dims=96, depths=[2,2,15,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") #GFlops:  8.71577472 Params:  50147752 826
        sv2 = partial(_model.VSSM, dims=96, depths=[2,2,20,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  8.612211455999999 Params:  49012840  1049
        bv0 = partial(_model.VSSM, dims=128, depths=[2,2,27,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v0", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1") # GFlops:  18.020115135999998 Params:  76254056 157
        bv1 = partial(_model.VSSM, dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") #GFlops:  15.358944256000001 Params:  88557800 606
        bv2 = partial(_model.VSSM, dims=128, depths=[2,2,20,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  15.220859904 Params:  86614504 774

        abt_tv0 = tv0
        # abt_tv0_re = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v01", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1") # GFlops:  5.62689504 Params:  22893448 407
        abt_tv0_csmtri = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v02", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1") # GFlops:  5.62689504 Params:  22893448 436
        abt_tv0_flex = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v04", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1") # GFlops:  5.62689504 Params:  22893448 432
        abt_tv0_noeinlayout = partial(_model.VSSM, dims=96, depths=[2,2,9,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, forward_type="v05", mlp_ratio=0.0, downsample_version="v1", patchembed_version="v1", norm_layer="ln2d") # GFlops:  5.62689504 Params:  22893448 594
        abt_tv1_mlp = partial(_model.VSSM, dims=96, depths=[2,2,2,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=-1,  forward_type="v05", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  5.632660223999999 Params:  28991656 773
        abt_tv1_161_2222_mlp = partial(_model.VSSM, dims=96, depths=[2,2,2,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1,  forward_type="v05", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # 
        abt_tv1_161 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1,  forward_type="v05", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  5.178818303999999 Params:  28256680 1079 #a8ln
        abt_tv1_161_noz = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1,  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.860903167999999 Params:  26247592 1113 #a9d
        abt_tv1_12_noz = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=-1,  forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.833356544 Params:  30633256 1290 
        abt_tv1_12_noz_dw = tv1

        abt_csm = tv2
        abt_ab1d = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.905609984 Params:  30249064 1601
        abt_ab2d = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v052d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.905609984 Params:  30249064 1600
        abt_ab2dc = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v052dc_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.905609984 Params:  30249064 1059

        abt_cv_3_false = tv2
        abt_cv_3_true = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=True, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.905609984 Params:  30254248 1574
        abt_cv_none = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=True, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.891385088 Params:  30202408 1615
        abt_cv_pos = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=-1, ssm_conv_bias=True, forward_type="v051d_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d",  posembed=True) # GFlops:  4.891385088 Params:  30503464 1605

        abt_init_mamba = tv2
        abt_init_rand = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_init="v1") # GFlops:  4.905609984 Params:  30249064 1593
        abt_init_zero = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_init="v2") # GFlops:  4.905609984 Params:  30249064 1591

        abt_dstate_1 = tv1
        abt_dstate_2 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=2, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.976960255999999 Params:  30802600 1198
        abt_dstate_4 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=4, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  5.215396607999999 Params:  30996136 1081
        abt_dstate_8 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=8, ssm_dt_rank="auto", ssm_ratio=1.5, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  5.044246272 Params:  28640008 1079
        abt_dstate_16 = partial(_model.VSSM, dims=96, depths=[2,2,5,2], ssm_d_state=16, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.8730959359999995 Params:  26283880 1094
        
        abt_mratio_4_8 = tv2 # GFlops:  4.905609984 Params:  30249064  1603
        abt_mratio_3_8 = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.5, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=3.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.6511423999999995 Params:  28485640 1340
        abt_mratio_2_8 = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.5, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=2.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.9514457599999995 Params:  2986688 1015

        abt_onorm_ln = tv2
        abt_onorm_none = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_onnone_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.897707264 Params:  30238696 1683
        abt_onorm_dwconv = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_ondwconv3_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.91193216 Params:  30285352 1613
        abt_onorm_oncnorm = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_oncnorm_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.91983488 Params:  30295720 1571
        abt_onrom_softmax = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_onsoftmax_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.897707264 Params:  30238696 1644
        abt_onrom_sigmoid = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_onsigmoid_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d") # GFlops:  4.897707264 Params:  30238696 1647

        abt_cact_silu = tv2
        abt_cact_gelu = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_act_layer="gelu") # GFlops:  4.905609984 Params:  30249064 1590
        abt_cact_relu = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_act_layer="relu") # GFlops:  4.905609984 Params:  30249064 1588
        abt_cact_sigmoid = partial(_model.VSSM, dims=96, depths=[2,2,8,2], ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=1.0, ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="ln2d", ssm_act_layer="sigmoid") # GFlops:  4.905609984 Params:  30249064 1584  

        size = 224
        for config in [
            tv0, sv0, bv0,
            tv1, sv1, bv1,
            tv2, sv2, bv2,
            abt_tv0, abt_tv0_csmtri, abt_tv0_flex, abt_tv0_noeinlayout,
            abt_tv1_mlp, abt_tv1_161, abt_tv1_161_noz, abt_tv1_12_noz, abt_tv1_12_noz_dw,
            abt_csm, abt_ab1d, abt_ab2d, abt_ab2dc,
            abt_cv_3_false, abt_cv_3_true, abt_cv_none, abt_cv_pos,
            abt_init_mamba, abt_init_rand, abt_init_zero,
            abt_dstate_1, abt_dstate_2, abt_dstate_4, abt_dstate_8, abt_dstate_16,
            abt_mratio_4_8, abt_mratio_3_8, abt_mratio_2_8,
            abt_onorm_ln, abt_onorm_none, abt_onorm_dwconv, abt_onorm_oncnorm, abt_onrom_softmax, abt_onrom_sigmoid,
            abt_cact_silu, abt_cact_gelu, abt_cact_relu, abt_cact_sigmoid,
        ]:
            print(get_variable_name(config, locals()), "============")
            Throughput.testall(config(), dataloader, args.data_path, size, args.batch_size)
        return

    return


if __name__ == "__main__":
    main()

    
