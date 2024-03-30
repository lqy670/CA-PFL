#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
from torch import nn
from typing import List, Tuple


__all__ = ["get_features"]


class Hook:
    def __init__(self):
        self.features = None
        pass

    def hook_fun(self, module, inputs, outputs):
        self.features = outputs


def register_for_hooks(network: nn.Module, registed_layers: List[str]):
    """
    采用pytorch中的hook方式获取每一层的激活输出
    :param network: 网络模型
    :param out_names: 获取哪些层的输出
    :return:
    """
    feature_hooks = []
    for n, m in network.named_modules():
        # 不存在钩子点，则直接挂载整个网络中每一层为钩子点
        if registed_layers is None and len(registed_layers) == 0:
            hook = Hook()
            m.register_forward_hook(hook.hook_fun)
            feature_hooks.append(hook)
        else:
            if n in registed_layers:
                hook = Hook()
                m.register_forward_hook(hook.hook_fun)
                feature_hooks.append(hook)
    return feature_hooks


def get_features(network: nn.Module, data: torch.Tensor, out_names: List[str]):
    """
    获取网络模型的中间输出结果
    :param network: 网络模型结构
    :param data: 执行前向运算的输出
    :param out_names: 提取哪些中间层的输出
    :return:
    """
    n = data.shape[0]
    # 调用函数，完成hook注册
    hooks = register_for_hooks(network, registed_layers=out_names)
    network.eval()
    with torch.no_grad():
        # 前向运算
        network(data)
    # return [hook.features.cpu().numpy() for hook in hooks]
    return [hook.features.reshape(n, -1).cpu().numpy() for hook in hooks]


def get_packets_from_net(ntlst: List[Tuple[str, torch.nn.Module]]):
    """
    将模型参数分成一个个的packets, 即将卷积层与其相邻的BN等层合并在一起
    :param List[Tuple[str, torch.nn.Module]] ntlst:网络层名称与其模块的元组列表
    """
    # 合并conv layer + bn + relu等层
    pns = {}
    regarding = [
        torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
        torch.nn.LazyConv1d, torch.nn.LazyConv2d, torch.nn.LazyConv3d
    ]
    for name, module in ntlst:
        parent = ".".join(name.split(".")[:-1])
        for instance in regarding:
            if isinstance(module, instance):
                if parent not in pns.keys():
                    pns[parent] = [name]
                else:
                    pns[parent].append(name)
                pass
    return pns


if __name__ == "__main__":
    from network.simple_cnn import SimpleCNN
    from network.resnet import resnet9
    from utils.net_ops import NetOps
    # net = SimpleCNN()
    net = resnet9()
    ops = NetOps(net)
    for k, v in ops.ntlst:
        print(k, v)

    x = torch.randn([50, 3, 32, 32])
    packets = get_packets_from_net(ops.ntlst)
    for pk, pv in packets.items():
        print(pk, pv)
    
    rs = get_outputs_from_specific_layer(net, x, out_names=list(packets.keys()))
    for r in rs:
        print(r.shape)
    pass

