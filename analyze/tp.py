import time
import torch
import torch.utils.data
import argparse
import os
import sys
import logging
from torchvision import datasets, transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.models.vision_transformer import EncoderBlock
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

HOME = os.environ["HOME"].rstrip("/")


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module



def get_dataloader(batch_size=64, root="./val", img_size=224):
    size = int((256 / 224) * img_size)
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])

    dataset = datasets.ImageFolder(root, transform=transform)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )
    return data_loader


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        torch.cuda.reset_peak_memory_stats()
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return


def testfwdbwd(data_loader, model, logger):
    model.train()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images).sum().backward()
        torch.cuda.synchronize()
        logger.info(f"testfwdbwd averaged with 30 times")
        torch.cuda.reset_peak_memory_stats()
        tic1 = time.time()
        for i in range(30):
            model(images).sum().backward()
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} testfwdbwd {30 * batch_size / (tic2 - tic1)}")
        logger.info(f"batch_size {batch_size} mem cost {torch.cuda.max_memory_allocated() / 1024 / 1024} MB")
        return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--size', type=int, default=224, help='path to dataset')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataloader = get_dataloader(
        batch_size=args.batch_size, 
        root=os.path.join(os.path.abspath(args.data_path), "val"),
        img_size=args.size,
    )

    # vim: install mamba_ssm
    if False:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/Vim/mamba-1p1p1")
        sys.path.insert(0, specpath)
        model = import_abspy("models_mamba", f"{HOME}/OTHERS/Vim/vim")
        model = model.vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
        model.cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging)
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]
        """
        25796584
        INFO:root:batch_size 128 throughput 808.9786050307463
        INFO:root:batch_size 128 mem cost 1054.6875 MB
        INFO:root:batch_size 128 testfwdbwd 231.73345700794215
        INFO:root:batch_size 128 mem cost 16141.994140625 MB
        """
    
    # convnext-s4nd: this needs timm=0.5.4; install extentions/kernel
    if False:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./convnexts4nd")
        sys.path.insert(0, specpath)
        model = import_abspy("convnext_timm", os.path.join(os.path.dirname(__file__), "./convnexts4nd"))
        model = model.convnext_tiny_s4nd().cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging) 
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]
        """
        30007057
        INFO:root:batch_size 128 throughput 682.8001707792412
        INFO:root:batch_size 128 mem cost 3945.23486328125 MB
        INFO:root:batch_size 128 testfwdbwd 254.2786610643497
        INFO:root:batch_size 128 mem cost 24629.96728515625 MB
        """

    # swin: install kernels/window_process
    if False:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/Swin-Transformer")
        sys.path.insert(0, specpath)
        _model = import_abspy("swin_transformer", f"{HOME}/OTHERS/Swin-Transformer/models")
        # configs/swin/swin_tiny_patch4_window7_224.yaml
        model = _model.SwinTransformer(embed_dim=96, depths=[2,2,6,2], num_heads=[ 3, 6, 12, 24 ], fused_window_process=True)
        # configs/swin/swin_small_patch4_window7_224.yaml
        model = _model.SwinTransformer(embed_dim=96, depths=[2,2,18,2], num_heads=[ 3, 6, 12, 24 ], fused_window_process=True)
        # # configs/swin/swin_base_patch4_window7_224.yaml
        model = _model.SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[ 4, 8, 16, 32 ], fused_window_process=True)
        model.cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging)
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]
        """
        28288354
        INFO:root:batch_size 128 throughput 1240.4891151522322
        INFO:root:batch_size 128 mem cost 2254.42333984375 MB
        INFO:root:batch_size 128 testfwdbwd 365.19026815764215
        INFO:root:batch_size 128 mem cost 13423.412109375 MB
        49606258
        INFO:root:batch_size 128 throughput 716.4445344379523
        INFO:root:batch_size 128 mem cost 2336.19091796875 MB
        INFO:root:batch_size 128 testfwdbwd 209.40260390195903
        INFO:root:batch_size 128 mem cost 21352.494140625 MB
        87768224
        INFO:root:batch_size 128 throughput 457.88358249876234
        INFO:root:batch_size 128 mem cost 3166.45654296875 MB
        INFO:root:batch_size 128 testfwdbwd 134.25118787105706
        INFO:root:batch_size 128 mem cost 28620.9677734375 MB
        """

    # convnext:
    if False:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/Swin-Transformer")
        sys.path.insert(0, specpath)
        _model = import_abspy("convnext", f"{HOME}/OTHERS/ConvNeXt/models")
        model = _model.convnext_tiny()
        model = _model.convnext_small()
        model = _model.convnext_base()
        model.cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging)
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]
        """
        28589128
        INFO:root:batch_size 128 throughput 1175.4456955462083
        INFO:root:batch_size 128 mem cost 1664.06689453125 MB
        INFO:root:batch_size 128 testfwdbwd 366.9606466184614
        INFO:root:batch_size 128 mem cost 14552.92041015625 MB
        50223688
        INFO:root:batch_size 128 throughput 676.5422186351589
        INFO:root:batch_size 128 mem cost 1745.59619140625 MB
        INFO:root:batch_size 128 testfwdbwd 208.60340091312548
        INFO:root:batch_size 128 mem cost 22694.73681640625 MB
        88591464
        INFO:root:batch_size 128 throughput 433.2247731814844
        INFO:root:batch_size 128 mem cost 2380.07470703125 MB
        INFO:root:batch_size 128 testfwdbwd 132.96389179674586
        INFO:root:batch_size 128 mem cost 30388.55419921875 MB
        """

    # hivit:
    if False:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/Swin-Transformer")
        sys.path.insert(0, specpath)
        _model = import_abspy("hivit", f"{HOME}/OTHERS/hivit/supervised/models/")
        model = _model.HiViT(patch_size=16, inner_patches=4, embed_dim=384, depths=[1, 1, 10], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        model = _model.HiViT(patch_size=16, inner_patches=4, embed_dim=384, depths=[2, 2, 20], num_heads=6, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        model = _model.HiViT(patch_size=16, inner_patches=4, embed_dim=512, depths=[2, 2, 20], num_heads=8, stem_mlp_ratio=3., mlp_ratio=4., ape=True, rpe=True,)
        model.cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging)
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]
        """
        19181668
        INFO:root:batch_size 128 throughput 1365.9227080098276
        INFO:root:batch_size 128 mem cost 1337.55224609375 MB
        INFO:root:batch_size 128 testfwdbwd 444.17530196911287
        INFO:root:batch_size 128 mem cost 11668.7783203125 MB
        37526464
        INFO:root:batch_size 128 throughput 704.4131339427528
        INFO:root:batch_size 128 mem cost 1410.47412109375 MB
        INFO:root:batch_size 128 testfwdbwd 229.196284317322
        INFO:root:batch_size 128 mem cost 22387.921875 MB
        66418952
        INFO:root:batch_size 128 throughput 455.69311730304236
        INFO:root:batch_size 128 mem cost 1913.41357421875 MB
        INFO:root:batch_size 128 testfwdbwd 146.59518722685985
        INFO:root:batch_size 128 mem cost 29929.88720703125 MB
        """

    # internimage: install classification/ops_dcnv3
    if False:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
        sys.path.insert(0, specpath)
        _model = import_abspy("intern_image", f"{HOME}/OTHERS/InternImage/classification/models/")
        model = _model.InternImage(core_op='DCNv3', channels=64, depths=[4, 4, 8, 4], groups=[4, 8, 16, 32], offset_scale=1.0, mlp_ratio=4.,)
        model = _model.InternImage(core_op='DCNv3', channels=80, depths=[4, 4, 21, 4], groups=[5, 10, 20, 40], layer_scale=1e-5, offset_scale=1.0, mlp_ratio=4., post_norm=True)
        model = _model.InternImage(core_op='DCNv3', channels=112, depths=[4, 4, 21, 4], groups=[7, 14, 28, 56], layer_scale=1e-5, offset_scale=1.0, mlp_ratio=4., post_norm=True)
        model.cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging)
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]
        """
        22195704
        INFO:root:batch_size 128 throughput 978.9873366017044
        INFO:root:batch_size 128 mem cost 1343.203125 MB
        INFO:root:batch_size 128 testfwdbwd 295.5451528170684
        INFO:root:batch_size 128 mem cost 17261.25341796875 MB
        50079880
        INFO:root:batch_size 128 throughput 559.2031560390494
        INFO:root:batch_size 128 mem cost 1622.33642578125 MB
        INFO:root:batch_size 128 testfwdbwd 154.76046438534033
        INFO:root:batch_size 128 mem cost 31786.06884765625 MB
        97461832
        INFO:root:batch_size 128 throughput 370.07997510699914
        INFO:root:batch_size 128 mem cost 2343.810546875 MB
        INFO:root:batch_size 128 testfwdbwd 101.8028548882445
        INFO:root:batch_size 128 mem cost 44547.39697265625 MB
        """

    # vssm: install selective_scan
    if True:
        specpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{HOME}/OTHERS/InternImage/classification")
        sys.path.insert(0, specpath)
        _model = import_abspy("vmamba", f"../VMamba/classification/models")
        model = _model.VSSM(dims=96,)
        model.cuda().eval()
        print(parameter_count(model)[""])
        throughput(data_loader=dataloader, model=model, logger=logging)
        testfwdbwd(data_loader=dataloader, model=model, logger=logging)
        sys.path = sys.path[1:]



    
