#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from .simple_cnn import SimpleCNN
from .shufflenet import shufflenet
from .resnet import resnet9, resnet18, resnet34, resnet50, resnet101, resnet152
from .moon_net import moon_resnet9, moon_resnet18, moon_resnet34, moon_resnet50, \
    moon_resnet101, moon_resnet152, moon_shufflenet, moon_SimpleCNN, ModelFedCon
from .charrnn import chargru, charlstm
from .cpfl_net import cpfl_resnet9, cpfl_resnet18, cpfl_resnet34, cpfl_resnet50, \
    cpfl_resnet101, cpfl_resnet152, cpfl_SimpleCNN, Modelcpfl
