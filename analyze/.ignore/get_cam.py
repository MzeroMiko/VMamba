from get_scaleup import build_loader_val, build_vssm, build_mmpretrain_models
from vmamba.vmamba import VSSM, VSSLayer, VSSBlock, SS2D

model = build_vssm(ckpt="/home/LiuYue/Workspace3/ckpts/ckpt_vssm_tiny_224/ckpt_epoch_292.pth")
data_loader_val = build_loader_val(batch_size=1, sequential=True)
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD 
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


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

model = add_hook(model)

data = next(iter(data_loader_val))

def denormalize(image: torch.Tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    C, H, W = image.shape
    image = image.cpu() * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    image = image.permute(1, 2, 0).numpy()
    # image = (image.permute(1, 2, 0) * 255).to(torch.uint8).numpy()
    return image

rgb_img = denormalize(data[0][0])
inputs = data[0].cuda().requires_grad_(True)

simpnorm = lambda x: (x - x.min()) / (x.max() - x.min())
out = model(inputs)
# target = (1  - out.softmax(dim=-1)[data[1]]).sum() + out.sum() * 0.0
# grad = torch.autograd.grad(target, inputs)
# print(len(grad), grad[0].shape)

# simpnorm = lambda x: (x - x.min()) / (x.max() - x.min())
# grayscale_cam = simpnorm(grad[0][0].permute(1, 2, 0)).max(dim=-1)[0].cpu().numpy() 
grayscale_cam = getattr(model.layers[-2].blocks[-1].self_attention, "y1")
grayscale_cam += getattr(model.layers[-2].blocks[-1].self_attention, "y2")
grayscale_cam += getattr(model.layers[-2].blocks[-1].self_attention, "y3")
grayscale_cam += getattr(model.layers[-2].blocks[-1].self_attention, "y4")
grayscale_cam = grayscale_cam.view(-1, 14, 14).permute(1, 2, 0)
grayscale_cam = simpnorm(grayscale_cam.sum(dim=-1))
grayscale_cam = F.interpolate(grayscale_cam[None, None],(224, 224), mode="bicubic")[0, 0].detach().cpu().numpy()
print(grayscale_cam.shape)

visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
from PIL import Image
Image.fromarray(visualization).save("1.jpg")
exit()





# ===========================================


target_layers = [model.layers[-1].blocks[-1].self_attention.out_norm]
input_tensor = data[0]
cam = GradCAM(model=model, target_layers=target_layers)
targets = [ClassifierOutputTarget(data[1])]
grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
model_outputs = cam.outputs
from matplotlib import pyplot as plt
print(visualization.dtype, visualization.shape)
from PIL import Image
Image.fromarray(visualization).save("1.jpg")

