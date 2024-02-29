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
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to dataset')
    parser.add_argument('--size', type=int, default=224, help='path to dataset')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataloader = get_dataloader(
        batch_size=args.batch_size, 
        root=os.path.join(os.path.abspath(args.data_path), "val"),
        img_size=args.size,
    )

    MODEL = "VSSM"

    if MODEL in ["VSSM"]:
        VSSM = import_abspy(
            "vmamba", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../VMamba/classification/models"),
        ).VSSM
        model = VSSM(dims=96, depths=[2,2,5,2], ssm_d_state=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 390.44661250848594
        model = VSSM(dims=96, depths=[2,2,15,2], ssm_d_state=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 245.8051057770092
        model = VSSM(dims=128, depths=[2,2,15,2], ssm_d_state=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 175.17029874793926
        model = VSSM(dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_ratio=1, forward_type="v3noz", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 383.81260980073216 # A100
        model = VSSM(dims=128, depths=[2,2,15,2], ssm_d_state=1, ssm_ratio=1, forward_type="dev", downsample_version="v3", patchembed_version="v2")
        # INFO:root:batch_size 64 throughput 409.4404580002472 # A100
        
    if MODEL in ["SWIN"]:
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Swin-Transformer"),)
        # print(sys.path)
        # from kernels.window_process.window_process import WindowProcess, WindowProcessReverse
        SWIN = import_abspy(
            "swin_transformer", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Swin-Transformer/models"),
        ).SwinTransformer
        model = SWIN()
        # INFO:root:batch_size 64 throughput 748.1943721697307
        model = SWIN(embed_dim=96, depths=[2,2,18,2])
        # INFO:root:batch_size 64 throughput 437.82169310230506
        model = SWIN(embed_dim=128, depths=[2,2,18,2], num_heads=[ 4, 8, 16, 32 ])
        # INFO:root:batch_size 64 throughput 284.90212452944024
        # INFO:root:batch_size 64 throughput 410.8976187260478 # A100

    if MODEL in ["CONVNEXT"]:
        CONVNEXT = import_abspy(
            "convnext", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../ConvNeXt/models"),
        )
        model = CONVNEXT.convnext_tiny()
        # INFO:root:batch_size 64 throughput 748.796215229091
        model = CONVNEXT.convnext_small()
        # INFO:root:batch_size 64 throughput 437.4729938751549
        model = CONVNEXT.convnext_base()
        # INFO:root:batch_size 64 throughput 284.73416141184316
    
    if MODEL in ["HIVIT"]:
        HIVIT = import_abspy(
            "models_hivit", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../hivit/self_supervised/models"),
        )
        model = HIVIT.hivit_base()
        # INFO:root:batch_size 128 throughput 301.77430420925174
    
    if MODEL in ["HEAT"]:
        HEAT = import_abspy(
            "heat", 
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"),
        ).HeatM
        model = HEAT(dims=64, depths=[4,4,16,4])
        # INFO:root:batch_size 64 throughput 428.96341939895416
        model = HEAT(dims=112, depths=[4,4,21,4])
        # INFO:root:batch_size 64 throughput 188.57667325700203

    model.cuda().eval()
    throughput(data_loader=dataloader, model=model, logger=logging)

