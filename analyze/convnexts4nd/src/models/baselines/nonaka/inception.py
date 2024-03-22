import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.baselines.nonaka.basic_conv1d import AdaptiveConcatPool1d, create_head1d

########################################################################################################
# Inception time inspired by https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py and https://github.com/tcapelle/TimeSeries_fastai/blob/master/inception.py

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    """Convolution with padding."""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def noop(x): return x

class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, act='linear', bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if (bottleneck_size>0) else noop

        self.convs = nn.ModuleList([conv(bottleneck_size if (bottleneck_size>0) else ni, nb_filters, ks) for ks in kss])
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss)+1)*nb_filters), nn.ReLU())

    def forward(self, x):
        #print("block in",x.size())
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs]+[self.conv_bottle(x)], dim=1))
        return out

class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn=nn.ReLU(True)
        self.conv=conv(ni, nf, 1)
        self.bn=nn.BatchNorm1d(nf)

    def forward(self, inp, out):
        #print("sk",out.size(), inp.size(), self.conv(inp).size(), self.bn(self.conv(inp)).size)
        #input()
        return self.act_fn(out + self.bn(self.conv(inp)))

class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()

        self.depth = depth
        assert((depth % 3) == 0)
        self.use_residual = use_residual

        n_ks = len(kss) + 1
        self.im = nn.ModuleList([InceptionBlock1d(input_channels if d==0 else n_ks*nb_filters,nb_filters=nb_filters,kss=kss, bottleneck_size=bottleneck_size) for d in range(depth)])
        self.sk = nn.ModuleList([Shortcut1d(input_channels if d==0 else n_ks*nb_filters, n_ks*nb_filters) for d in range(depth//3)])

    def forward(self, x):

        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = (self.sk[d//3])(input_res, x)
                input_res = x.clone()
        return x

class Inception1d(nn.Module):
    """Inception time architecture."""

    def __init__(self, num_classes=2, input_channels=8, kernel_size=40, depth=6, bottleneck_size=32, nb_filters=32, use_residual=True,lin_ftrs_head=None, ps_head=0.5, bn_final_head=False, bn_head=True, act_head="relu", concat_pooling=True):
        super().__init__()
        assert(kernel_size>=40)
        kernel_size = [k-1 if k%2==0 else k for k in [kernel_size,kernel_size//2,kernel_size//4]] #was 39,19,9

        layers = [InceptionBackbone(input_channels=input_channels, kss=kernel_size, depth=depth, bottleneck_size=bottleneck_size, nb_filters=nb_filters, use_residual=use_residual)]

        n_ks = len(kernel_size) + 1
        #head
        head = create_head1d(n_ks*nb_filters, nc=num_classes, lin_ftrs=lin_ftrs_head, ps=ps_head, bn_final=bn_final_head, bn=bn_head, act=act_head, concat_pooling=concat_pooling)
        layers.append(head)
        #layers.append(AdaptiveConcatPool1d())
        #layers.append(Flatten())
        #layers.append(nn.Linear(2*n_ks*nb_filters, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        y = self.layers(x.transpose(-1, -2))
        return y, None

    def get_layer_groups(self):
        depth = self.layers[0].depth
        if(depth>3):
            return ((self.layers[0].im[3:],self.layers[0].sk[1:]),self.layers[-1])
        else:
            return (self.layers[-1])

    def get_output_layer(self):
        return self.layers[-1][-1]

    def set_output_layer(self,x):
        self.layers[-1][-1] = x

def inception1d(**kwargs):
    """Constructs an Inception model."""
    return Inception1d(**kwargs)
