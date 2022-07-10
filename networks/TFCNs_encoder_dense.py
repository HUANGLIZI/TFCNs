import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            #we pass all previous activations into each dense layer normally
            #But we only store each dense layer's output in the new_features array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) 
            return x

class CLAB_BLOCK(nn.Module):
    def __init__(self,in_channels,p_block,k_size,size):
        super(CLAB_BLOCK, self).__init__()
        self.size = size
        self.layers = nn.ModuleList([nn.Conv2d(in_channels, k_size, kernel_size=1,
                                          stride=1, padding=0, bias=True) for i in range(p_block)])
        self.conv_sig = nn.Sequential(
            nn.Conv2d(p_block,in_channels , kernel_size=1,stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        self.ln = Linear(p_block,p_block)
        self.norm = nn.BatchNorm2d(num_features = 1)  

    def forward(self,x):
        x_temp = []
        for layer in self.layers:
            branch = layer(x)
            branch = self.relu(branch)
            branch = torch.mean(branch,dim=1)
            b,h,w = branch.size()
            branch = branch.view(b,1,h,w)
            branch = self.norm(branch)
            x_temp.append(branch)
        output = torch.cat(x_temp,1)
        output = nn.AvgPool2d(kernel_size=self.size, stride=1, padding=0)(output)
        b,c,h,w = output.size()
        output = output.view(b,1,c)
        output = self.ln(output)
        output = output.view(b,c,h,w)
        output = self.conv_sig(output)
        output = output * x
        return output



class TransitionDown(nn.Sequential):
    def __init__(self, in_channels,odd):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(0.2))
        if odd == 0:
            self.add_module('maxpool', nn.MaxPool2d(2))
        else:
            self.add_module('maxpool', nn.MaxPool2d(2,padding=1))

    def forward(self, x):
        return super().forward(x)

class Encoder_Dense(nn.Module):
    """Implementation of part of encoder"""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        self.clab_0 = CLAB_BLOCK(64,16,5,112)
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', DenseBlock(width,64,3))] 
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', TransitionDown(width*4,1))] +
                [('unit2', DenseBlock(width*4,64,4))],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', TransitionDown(width*8,0))] +
                [('unit2', DenseBlock(width*8,64,8)) ],
                ))),
        ]))
        self.clab_set = nn.Sequential(OrderedDict([('cuab1',CLAB_BLOCK(256,16,5,55)),('cuab2',CLAB_BLOCK(512,16,5,28))]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        x = self.clab_0(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            x = self.clab_set[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]
