import os
import torch
import torch.nn.functional as F
from vmamba.vmamba import VSSM, VSSLayer, VSSBlock, SS2D
from functools import partial
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from matplotlib import pyplot
from PIL import Image
import math

ckpt = "/home/LiuYue/Workspace3/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth"
data = "/dataset/ImageNet2012/"


def add_hook(model: VSSM):

    def ss2d_forward(self: SS2D, x: torch.Tensor = None):
        if True:
            B, H, W, C = x.shape

            xz = self.in_proj(x)
            x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x)) # (b, d, h, w)
            y1, y2, y3, y4 = self.forward_core(x)
            assert y1.dtype == torch.float32
            
            setattr(self, "y1", y1)
            setattr(self, "y2", y2)
            setattr(self, "y3", y3)
            setattr(self, "y4", y4)
            y = y1 + y2 + y3 + y4
            y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = self.out_norm(y)
            y = y * F.silu(z)

            setattr(self, "yz", y.view(B, H * W, -1).permute(0, 2, 1))
            
            out = self.out_proj(y)
            if self.dropout is not None:
                out = self.dropout(out)
            return out

    for layer in model.layers:
        for blk in layer.blocks:
            ss2d = blk.self_attention
            setattr(ss2d, "DEBUG", True)
            ss2d.forward = partial(ss2d_forward, ss2d)

    return model


def build_dataset(root=data):
    root = os.path.join(root, "val")
    transform = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=InterpolationMode.BICUBIC),
        # transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset


def denormalize(image: torch.Tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    C, H, W = image.shape
    image = image.cpu() * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    image = Image.fromarray(image)
    return image


def show_attn(y: torch.Tensor):
    B, C, L = y.shape
    H = int(math.sqrt(L))
    W = H
    # y = y.softmax(dim=-1)
    # y = y.log()
    # y = y / y.sum(dim=-1).view(B, C, 1)
    y = y.permute(0, 2, 1).view(B, H, W, C)
    y = y.mean(dim=-1, keepdims=True)[0].repeat(1, 1, 3)
    # y = y.max(dim=-1, keepdims=True)[0][0].repeat(1, 1, 3)
    y = ((y - y.min()) / (y.max() - y.min()))
    y = (y * 255).cpu().to(torch.uint8).numpy()
    return y

dataset = build_dataset()
target_index = dataset.class_to_idx["n01737021"]

# n01737021 water snake # 2903...
# n01749939 green mamba # 3203...
# for i, d in enumerate(dataset):
#     if d[1] == target_index:
#         print(i, d[0])

# image = denormalize(dataset[3213][0])
# image = denormalize(dataset[2907][0])
image = denormalize(dataset[2907][0])
os.makedirs(os.path.dirname("show/ori/1.jpg"), exist_ok=True)
os.makedirs(os.path.dirname("show/feats/1.jpg"), exist_ok=True)
image.save("show/ori/1.jpg")

# n01749939 green mamba

input = dataset[2907][0][None, :, :, :].cuda().half()

model = VSSM(depths=[2,2,9,2])
model.load_state_dict(torch.load(open(ckpt, "rb"))['model'])
model = model.half().cuda()
model.eval()
model = add_hook(model)

with torch.cuda.amp.autocast():
    model(input)

for i, layer in enumerate(model.layers):
    for j, blk in enumerate(layer.blocks):
        ss2d = blk.self_attention
        ys = []
        for k, y in enumerate(["y1", "y2", "y3", "y4","yz"]):
            _y = getattr(ss2d, y, None)[0][None]
            ys.append(_y)
            Image.fromarray(show_attn(_y)).save(f"show/feats/{i}_{j}_{k}.jpg")
        Image.fromarray(show_attn(sum(ys[:4]))).save(f"show/feats/{i}_{j}_all.jpg")

