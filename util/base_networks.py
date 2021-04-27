""" This code was adapted from: https://github.com/jeffreyhuang1/two-stream-action-recognition/blob/master/network.py
"""

import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, nb_classes=101, channel=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1_custom = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3,   
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_custom = nn.Linear(512 * block.expansion, nb_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc_custom(x)
        return out


def resnet18(nb_classes, pretrained=False, channel=20, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet18'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet34(nb_classes, pretrained=False, channel=20, **kwargs):

    model = ResNet(BasicBlock, [3, 4, 6, 3], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet34'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet50(nb_classes, pretrained=False, channel=20, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet50'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)
    return model


def resnet101(nb_classes, pretrained=False, channel=20, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3], nb_classes=nb_classes, channel=channel, **kwargs)
    if pretrained:
       pretrain_dict = model_zoo.load_url(model_urls['resnet101'])                  # modify pretrain code
       model_dict = model.state_dict()
       model_dict=weight_transform(model_dict, pretrain_dict, channel)
       model.load_state_dict(model_dict)

    return model

def resnet152(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def cross_modality_pretrain(conv1_weight, channel):
    # transform the original 3 channel weight to "channel" channel
    S=0
    for i in range(3):
        S += conv1_weight[:,i,:,:]
    avg = S/3.
    new_conv1_weight = torch.FloatTensor(64,channel,7,7)
    #print type(avg),type(new_conv1_weight)
    for i in range(channel):
        new_conv1_weight[:,i,:,:] = avg.data
    return new_conv1_weight

def weight_transform(model_dict, pretrain_dict, channel):
    weight_dict  = {k:v for k, v in pretrain_dict.items() if k in model_dict}
    #print pretrain_dict.keys()
    w3 = pretrain_dict['conv1.weight']
    #print type(w3)
    if channel == 3:
        wt = w3
    else:
        wt = cross_modality_pretrain(w3,channel)

    weight_dict['conv1_custom.weight'] = wt
    model_dict.update(weight_dict)
    return model_dict







#################### MY STUFF ####################

class Conv2d_GN_ReLU(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLU, self).__init__()
        padding = 0 if ksize < 2 else ksize//2
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=ksize, stride=stride, 
                               padding=padding, bias=False)
        self.gn1 = nn.GroupNorm(num_groups, out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        return out

class Conv2d_GN_ReLUx2(nn.Module):
    """ Implements a module that performs 
            conv2d + groupnorm + ReLU + 
            conv2d + groupnorm + ReLU
            (and a possible downsampling operation)

        Assumes kernel size is odd
    """

    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Conv2d_GN_ReLUx2, self).__init__()
        self.layer1 = Conv2d_GN_ReLU(in_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)
        self.layer2 = Conv2d_GN_ReLU(out_channels, out_channels, 
                                     num_groups, ksize=ksize, stride=stride)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out

class Upsample_Concat_Conv2d_GN_ReLU(nn.Module):
    """ Implements a module that performs
            Upsample (reduction: conv2d + groupnorm + ReLU + bilinear_sampling) +
            concat + conv2d + groupnorm + ReLU 

        The Upsample operation consists of a Conv2d_GN_ReLU that reduces the channels by 2,
            followed by bilinear sampling

        Note: in_channels is number of channels of ONE of the inputs to the concatenation

    """
    def __init__(self, in_channels, out_channels, num_groups, ksize=3, stride=1):
        super(Upsample_Concat_Conv2d_GN_ReLU, self).__init__()
        self.channel_reduction_layer = Conv2d_GN_ReLU(in_channels, in_channels//2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = Conv2d_GN_ReLU(in_channels, out_channels, num_groups)

    def forward(self, x1, x2):
        x1 = self.channel_reduction_layer(x1)
        x1 = self.upsample(x1)
        out = torch.cat([x1, x2], dim=1) # Concat on channels dimension
        out = self.conv_gn_relu(out)

        return out

class Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch(nn.Module):
    """ Implements a module that performs
            Upsample (reduction: conv2d + groupnorm + ReLU + bilinear_sampling) +
            concat + conv2d + groupnorm + ReLU 
        for the U-Net decoding architecture with an arbitrary number of encoders

        The Upsample operation consists of a Conv2d_GN_ReLU that reduces the channels by 2,
            followed by bilinear sampling

        Note: in_channels is number of channels of ONE of the inputs to the concatenation

    """
    def __init__(self, in_channels, out_channels, num_groups, num_encoders, ksize=3, stride=1):
        super(Upsample_Concat_Conv2d_GN_ReLU_Multi_Branch, self).__init__()
        self.channel_reduction_layer = Conv2d_GN_ReLU(in_channels, in_channels//2, num_groups)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_gn_relu = Conv2d_GN_ReLU(int(in_channels//2 * (num_encoders+1)), out_channels, num_groups)

    def forward(self, x, skips):
        """ Forward module

            @param skips: a list of intermediate skip-layer torch tensors from each encoder
        """
        x = self.channel_reduction_layer(x)
        x = self.upsample(x)
        out = torch.cat([x] + skips, dim=1) # Concat on channels dimension
        out = self.conv_gn_relu(out)

        return out

def maxpool2x2(input, ksize=2, stride=2):
    """2x2 max pooling"""
    return nn.MaxPool2d(ksize, stride=stride)(input)


