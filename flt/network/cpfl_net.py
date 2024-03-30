#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import nn
from .simple_cnn import ConvBN
from .resnet import BasicBlock, BottleNeck


class ResNet_for_cpfl(nn.Module):
    def __init__(self, block, num_block, num_classes=10, in_channel: int = 3):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avg_pool(out)
        out = out.squeeze()
        y = self.fc(out)
        return out, out, y


def cpfl_resnet9(num_classes: int = 10, in_channel: int = 3):
    """ return a ResNet 9 object """
    return ResNet_for_cpfl(BasicBlock, [1, 1, 1, 1], num_classes=num_classes, in_channel=in_channel)


def cpfl_resnet18(num_classes: int = 10, in_channel: int = 3):
    """ return a ResNet 18 object """
    return ResNet_for_cpfl(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channel=in_channel)


def cpfl_resnet34(num_classes: int = 10, in_channel: int = 3):
    """ return a ResNet 34 object """
    return ResNet_for_cpfl(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channel=in_channel)


def cpfl_resnet50(num_classes: int = 10, in_channel: int = 3):
    """ return a ResNet 50 object """
    return ResNet_for_cpfl(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, in_channel=in_channel)


def cpfl_resnet101(num_classes: int = 10, in_channel: int = 3):
    """ return a ResNet 101 object """
    return ResNet_for_cpfl(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, in_channel=in_channel)


def cpfl_resnet152(num_classes: int = 10, in_channel: int = 3):
    """ return a ResNet 152 object """
    return ResNet_for_cpfl(BottleNeck, [3, 8, 36, 3], num_classes=num_classes, in_channel=in_channel)


class SimpleCNN_for_cpfl(nn.Module):
    def __init__(self, out_num: int = 10, in_channel: int = 3):
        super().__init__()
        self.block_1 = ConvBN(in_channel, 16, 3, 1, 1)

        self.block_2 = nn.Sequential(
            ConvBN(16, 16, 1, 1),
            ConvBN(16, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_3 = nn.Sequential(
            ConvBN(16, 32, 1, 1),
            ConvBN(32, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.block_4 = nn.Sequential(
            ConvBN(32, 64, 1, 1),
            ConvBN(64, 64, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block_5 = nn.Sequential(
            ConvBN(64, 128, 1, 1),
            ConvBN(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.block_6 = nn.Sequential(
            ConvBN(128, 256, 1, 1),
            ConvBN(256, 512, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_num)
        )

    def forward(self, x):
        h = self.block_1(x)
        h = self.block_2(h)
        h = self.block_3(h)
        h = self.block_4(h)
        h = self.block_5(h)
        h = self.block_6(h)
        h = h.squeeze()
        y = self.fc(h)
        return h, h, y


def cpfl_SimpleCNN(num_classes: int = 10, in_channel: int = 3):
    return SimpleCNN_for_cpfl(out_num=num_classes, in_channel=in_channel)


# projection、predictor层使用以下MLP结构(参考byol)
class MLP_projection(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLP_projection, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class MLP_predictor(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLP_predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)


class Modelcpfl(nn.Module):
    def __init__(self, model_name: str, num_classes: int, in_channel: int):
        super().__init__()
        if model_name == "resnet9":
            basemodel = cpfl_resnet9(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet18":
            basemodel = cpfl_resnet18(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet34":
            basemodel = cpfl_resnet34(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet50":
            basemodel = cpfl_resnet50(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet101":
            basemodel = cpfl_resnet101(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "resnet152":
            basemodel = cpfl_resnet152(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        elif model_name == "SimpleCNN":
            basemodel = cpfl_SimpleCNN(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512
        else:
            basemodel = cpfl_SimpleCNN(num_classes=num_classes, in_channel=in_channel)
            self.features = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = 512

        self.projection = MLP_projection(in_channels=num_ftrs, mlp_hidden_size=512, projection_size=128)
        self.predictor = MLP_predictor(in_channels=128, mlp_hidden_size=512, projection_size=128)
        self.l1 = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()
        x = self.projection(h)
        y = self.predictor(x)
        output = self.l1(y)
        return x, y, output
