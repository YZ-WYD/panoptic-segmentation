import math
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, relu_first=True, bias=False,
                 inplace=True):
        super(SeparableConv2d, self).__init__()
        self.relu_first = relu_first
        if self.relu_first:
            self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding=dilation,
                               dilation=dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        if self.relu_first:
            x = self.relu(x)
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, start_with_relu=True, grow_first=True,
                 is_last=False):
        super(Block, self).__init__()
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation))
            rep.append(nn.BatchNorm2d(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2))
            rep.append(nn.BatchNorm2d(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1))
            rep.append(nn.BatchNorm2d(planes))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        residual = x
        if self.skip is not None:
            residual = self.skip(x)
            residual = self.skipbn(residual)

        x = self.rep(x)
        x += residual
        return x


class Xception(nn.Module):
    def __init__(self, downsample_factor=16, pretrained=True, in_channels=3):
        super(Xception, self).__init__()
        if downsample_factor == 8:
            stride_list = [2, 1, 1]
        elif downsample_factor == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('downsample_factor should be 8 or 16')

        # 【修改】使用 in_channels
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        rate = 16 // downsample_factor
        self.block4 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 728, 3, stride_list[0], rate, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 1024, 2, stride_list[1], rate, start_with_relu=True, grow_first=False)
        self.block14 = Block(1024, 1536, 2, stride_list[2], rate, start_with_relu=True, grow_first=False)
        self.block15 = Block(1536, 2048, 2, 1, rate, start_with_relu=True, grow_first=False, is_last=True)

        self.block16 = Block(2048, 2048, 3, 1, rate, start_with_relu=True, grow_first=True)
        self.block17 = Block(2048, 2048, 3, 1, rate, start_with_relu=True, grow_first=True)
        self.block18 = Block(2048, 2048, 3, 1, rate, start_with_relu=True, grow_first=True)
        self.block19 = Block(2048, 2048, 3, 1, rate, start_with_relu=True, grow_first=True)
        self.block20 = Block(2048, 2048, 3, 1, rate, start_with_relu=True, grow_first=True)

        self._init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        low_level_features = self.block3(x)
        x = self.block4(low_level_features)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        return low_level_features, x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def xception(downsample_factor=16, pretrained=False, in_channels=3):
    model = Xception(downsample_factor, pretrained, in_channels)
    if pretrained:
        pretrained_dict = load_url(
            'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
            map_location='cpu')
        # 【修改】如果通道数不为3，跳过第一层（conv1）的权重
        if in_channels != 3:
            pretrained_dict.pop('conv1.weight', None)
        model.load_state_dict(pretrained_dict, strict=False)
    return model