import torch.nn as nn
import torch.nn.functional as F
import torch
import math


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class FPN(nn.Module):
    def __init__(self, in_feature_size=128, feature_size=256):
        super(FPN, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.prev_features_conv_512 = nn.Conv2d(feature_size * 8, feature_size, kernel_size=1, stride=1, padding=0)
        self.prev_features_conv_256 = nn.Conv2d(feature_size * 4, feature_size, kernel_size=1, stride=1, padding=0)
        self.prev_features_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.bn1 = nn.BatchNorm2d(feature_size)
        self.bn1_1 = nn.BatchNorm2d(feature_size)
        self.bn1_2 = nn.BatchNorm2d(feature_size)

        self.bn2 = nn.BatchNorm2d(feature_size * 2)
        self.bn2_1 = nn.BatchNorm2d(feature_size * 2)
        self.bn2_2 = nn.BatchNorm2d(feature_size * 2)

        self.bn3 = nn.BatchNorm2d(feature_size * 8)

        self.current_features_conv = nn.Conv2d(in_feature_size * 4, feature_size, kernel_size=1, stride=1, padding=0)

        self.forw_conf_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.forw_conf_1_1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.forw_conf_1_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.forw_conf_2 = nn.Conv2d(feature_size, feature_size * 2, kernel_size=3, stride=1, padding=1)
        self.forw_conf_2_1 = nn.Conv2d(feature_size * 2, feature_size * 2, kernel_size=3, stride=1, padding=1)
        self.forw_conf_2_2 = nn.Conv2d(feature_size * 2, feature_size * 2, kernel_size=3, stride=1, padding=1)

        self.forw_conf_3 = nn.Conv2d(feature_size * 2, feature_size * 8, kernel_size=1, stride=1, padding=0)

        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

    def forward(self, inputs):
        prev_features_512, prev_features_256, features = inputs

        prev_features_x_512 = self.prev_features_conv_512(prev_features_512)
        #prev_features_x_512 = self.prev_features_upsample(prev_features_x_512)

        prev_features_x_256 = self.prev_features_conv_256(prev_features_256)
        prev_features_x_256 = prev_features_x_256 + prev_features_x_512
        #prev_features_x_256 = self.prev_features_upsample(prev_features_x_256)

        features_x = self.current_features_conv(features)
        features_x = features_x + prev_features_x_256

        features_x = self.forw_conf_1(features_x)
        features_x = self.bn1(features_x)
        features_x = self.relu(features_x)

        features_x = self.forw_conf_1_1(features_x)
        features_x = self.bn1_1(features_x)
        features_x = self.relu(features_x)

        features_x = self.forw_conf_1_2(features_x)
        features_x = self.bn1_2(features_x)
        features_x = self.relu(features_x)

        features_x = self.forw_conf_2(features_x)
        features_x = self.bn2(features_x)
        features_x = self.relu(features_x)

        features_x = self.forw_conf_2_1(features_x)
        features_x = self.bn2_1(features_x)
        features_x = self.relu(features_x)

        features_x = self.forw_conf_2_2(features_x)
        features_x = self.bn2_2(features_x)
        features_x = self.relu(features_x)

        features_x = self.forw_conf_3(features_x)
        features_x = self.bn3(features_x)
        features_x = self.relu(features_x)

        return features_x


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers,
                 groups=1, width_per_group=64,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        self.hidden = None

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2),
                                       dilate=False)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=(1, 1),
                                       dilate=False)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=(1, 1),
                                       dilate=False)




        self.fpn = FPN()

        # self.output_pool = nn.MaxPool2d(4, 4)
        self.to_features = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0,
                                     bias=True)

        self.output_conv = nn.Conv2d(2048, 23, kernel_size=1, stride=1, padding=0,
                                     bias=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_1 = nn.Linear(2048, 58)

        if(True):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal(m.weight, mode='fan_out')
                elif isinstance(m, (nn.BatchNorm2d)):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, _norm_layer=None):
        if (_norm_layer == None):
            norm_layer = self._norm_layer
        else:
            norm_layer = _norm_layer

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            #downsample = True

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        #x_out = self.fpn([x4, x3, x2])

        x_out = self.to_features(x4)

        return x_out


def resnet():
    # resnet 50
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

