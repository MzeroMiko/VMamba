"""2D ResNet baselines from torchvision."""

import torch.nn as nn
import torchvision.models as models
from einops import rearrange


class TorchVisionResnet(nn.Module):
    def __init__(
        self,
        variant="resnet18",  # e.g. [ "resnet18" | "resnet34" | "resnet50" | "wide_resnet50_2" ]
    ):
        super().__init__()

        self.resnet = getattr(models, variant)(pretrained=False)

        # Remove pooling from stem: too much downsizing for CIFAR
        self.resnet.maxpool = nn.Identity()

        # Remove final head: handled by decoder
        self.d_output = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b ... h -> b h ...')
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.size(1) == 3:
            pass
        else:
            raise NotImplementedError
        y = self.resnet(x)
        return y, None
