#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class idenUnit(nn.Module):
    def __init__(self, input_channel, g):
        super(idenUnit, self).__init__()

        # bottle neck channel = input channel / 4, as the paper did
        neck_channel = int(input_channel / 4)

        # conv layers, GConv - (shuffle) -> DWConv -> Gconv
        #               bn, relu             bn        bn
        self.gconv1 = nn.Conv2d(input_channel, neck_channel, groups = g, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(neck_channel)

        self.dwconv = nn.Conv2d(neck_channel, neck_channel, groups = neck_channel, kernel_size = 3, 
            padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(neck_channel)

        self.gconv2 = nn.Conv2d(neck_channel, input_channel, groups = g, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(input_channel)

        # for channel shuffle operation 
        self.g, self.n = g, int(neck_channel/g)
        assert self.n == int(self.n), "wrong shape to shuffle"


    def forward(self, inputs):
        x = F.relu(self.bn1(self.gconv1(inputs)))
        
        # channel shuffle
        n, c, w, h = x.shape
        x = x.view(n, self.g, self.n, w, h) # type: ignore
        x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, w, h)

        x = self.bn(self.dwconv(x))
        x = self.bn2(self.gconv2(x))

        return F.relu(x + inputs)


class poolUnit(nn.Module):
    def __init__(self, input_channel, output_channel, g, first_group = True, downsample = True):
        super(poolUnit, self).__init__()
        self.downsample = downsample

        # bottle neck channel = input channel / 4, as the paper did
        neck_channel = int(output_channel / 4)

        # conv layers, GConv - (shuffle) -> DWConv -> Gconv
        #              bn,relu              bn        bn
        if first_group:
            self.gconv1 = nn.Conv2d(input_channel, neck_channel, groups = g, kernel_size = 1, bias = False)
        else:
            self.gconv1 = nn.Conv2d(input_channel, neck_channel, kernel_size = 1, bias = False)
        
        self.bn1 = nn.BatchNorm2d(neck_channel)

        stride = 2 if downsample else 1
        self.dwconv = nn.Conv2d(neck_channel, neck_channel, groups = neck_channel, stride = stride, kernel_size = 3, 
            padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(neck_channel)

        self.gconv2 = nn.Conv2d(neck_channel, output_channel - input_channel, groups = g, kernel_size = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channel - input_channel)

        # for channel shuffle operation 
        self.g, self.n = g, int(neck_channel/g)
        assert self.n == int(self.n), "error shape to shuffle"


    def forward(self, inputs):
        x = F.relu(self.bn1(self.gconv1(inputs)))
        
        # channel shuffle
        n, c, w, h = x.shape
        x = x.view(n, self.g, self.n, w, h)
        x = x.transpose_(1, 2).contiguous()
        x = x.view(n, c, w, h)

        x = self.bn(self.dwconv(x))
        x = self.bn2(self.gconv2(x))

        shortcut = F.avg_pool2d(inputs, 2) if self.downsample else inputs
        return F.relu(torch.cat((x, shortcut), dim = 1))


class ShuffleNet(nn.Module):
    def __init__(self, output_size, scale_factor = 1, g = 8):
        super(ShuffleNet, self).__init__()
        self.g = g
        # self.cs = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
        self.cs = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}

        # compute output channels for stages
        c2 = self.cs[self.g]
        c2 = int(scale_factor * c2)
        c3, c4 = 2*c2, 4*c2

        # first conv layer & last fc layer
        # self.conv1 = nn.Conv2d(3, 24, kernel_size = 3, padding = 1, stride = 1, bias = False)
        # self.bn1 = nn.BatchNorm2d(24)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size = 3, padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        # build stages
        self.stage2 = self.build_stage(24, c2, repeat_time = 3, first_group = False, downsample = False)
        self.stage3 = self.build_stage(c2, c3, repeat_time = 7)
        self.stage4 = self.build_stage(c3, c4, repeat_time = 3)
        
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(c4, output_size)

        # weights init
        self.weights_init()


    def build_stage(self, input_channel, output_channel, repeat_time, first_group = True, downsample = True):
        stage = [poolUnit(input_channel, output_channel, self.g, first_group = first_group, downsample = downsample)]
        
        for i in range(repeat_time):
            stage.append(idenUnit(output_channel, self.g)) # type: ignore

        return nn.Sequential(*stage) 



    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):

        # first conv layer
        # x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.stage1(inputs)
        # x = F.max_pool2d(x, kernel_size = 3, stride = 2, padding = 1)
        # assert x.shape[1:] == torch.Size([24,56,56])

        # bottlenecks
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # print(x.shape)

        # global pooling and fc (in place of conv 1x1 in paper)
        # x = F.adaptive_avg_pool2d(x, 1)
        x = self.adaptive_pool(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def shufflenet(num_classes: int = 10, g: int = 1, scale_factor: float = 0.5):
    return ShuffleNet(num_classes, g = g, scale_factor = scale_factor)


if __name__ == "__main__":
    import numpy as np
    x = np.random.randn(10, 3, 32, 32).astype(np.float32)
    x = torch.from_numpy(x)
    net = shufflenet()
    print(net)
    r = net(x)
    print(r)
