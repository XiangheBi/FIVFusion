# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models
import torch.nn.parallel
import torch.nn.parallel
import torch.nn.modules
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from collections import OrderedDict
import pytorch_ssim
from exclusion_loss import my_compute_gradient,compute_gradient_img_my,Scharr_demo
from torchvision.utils import save_image

class LaplacianLayer(nn.Module):
    def __init__(self, in_channels):
        super(LaplacianLayer, self).__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.kernel = kernel.repeat(in_channels, 1, 1, 1)

    def forward(self, x):
        self.kernel = self.kernel.to(x.device)
        gradient_orig = F.conv2d(x, self.kernel, stride=1, padding=1, groups=x.size(1))
        gradient_orig = torch.abs(gradient_orig)
        grad_min = gradient_orig.min()
        grad_max = gradient_orig.max()
        grad_norm = (gradient_orig - grad_min) / (grad_max - grad_min + 0.0001)
        return grad_norm
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        return x * scale
class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        self.laplaci = LaplacianLayer(64)

        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.laplaci(x) + x_out
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
def swish(x):
    return x * F.sigmoid(x)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class GlobalPoolStripAttention(nn.Module):
    def __init__(self, k) -> None:
        super().__init__()
        return x * self.beta + vert_out * self.gamma

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        m['bn1'] = nn.BatchNorm2d(n)
        m['ReLU1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(n, n, k, stride=s, padding=1)
        m['bn2'] = nn.BatchNorm2d(n)
        self.group1 = nn.Sequential(m)
        self.relu = nn.Sequential(nn.ReLU(inplace=True))
    def forward(self, x):
        out = self.group1(x) + x
        out = self.relu(out)
        return out
class residualBlock2(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock2, self).__init__()
    def forward(self, x):
        return out

class residualBlock3(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock3, self).__init__()
    def forward(self, x):
        return out

class model(nn.Module):
    def __init__(self, n_residual_blocks):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.n_residual_blocks = n_residual_blocks
        self.n_residual_blocks2 = 1
        self.n_residual_blocks3 = 1

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())
        for i in range(self.n_residual_blocks2):
            self.add_module('residual_block2' + str(i + 1), residualBlock2())
        for i in range(self.n_residual_blocks3):
            self.add_module('residual_block3' + str(i + 1), residualBlock3())
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        img = x[:,0,:,:].squeeze(1)
        H, W = x.size()[2], x.size()[3]
        im_gred = Scharr_demo(img.cpu().numpy())
        im_gred = Variable(torch.from_numpy(im_gred)).float().cuda()/255
        im_gred = im_gred.repeat(1,64,1,1)
        x = swish(self.relu1(self.bn1(self.conv1(x))))
        x = swish(self.relu2(self.bn2(self.conv2(x))))
        x = swish(self.relu3(self.bn3(self.conv3(x))))
        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)
            y = y+ im_gred*1
        x = swish(self.relu4(self.bn4(self.conv4(y)))) + x
        x1 = swish(self.relu5(self.bn5(self.conv5(x))))
        x2 = swish(self.relu7(self.bn7(self.conv7(x))))
        y1 = x1.clone()
        for i in range(self.n_residual_blocks2):
            y1 = self.__getattr__('residual_block2' + str(i + 1))(y1)
            y1 = y1+ im_gred*1
        y2 = x2.clone()
        for i in range(self.n_residual_blocks3):
            y2 = self.__getattr__('residual_block3' + str(i + 1))(y2)
            y2 = y2+ im_gred*1

        t1 = self.sigmoid(self.conv6(y1))
        t2 = self.sigmoid(self.conv8(y2))
        return t1 , t2
