#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import logging
import operator
import numpy as np
from torch import optim
from torch.utils import data


class FedBase(object):
    def __init__(
        self,
        global_net,
        nets: dict,
        datasets: dict,
        test_dataset: dict,
        nk_parties: int,
        E: int,
        comm_round: int,
        lr: float,
        batch_size: int,
        weight_decay: float,
        optim_name: str,
        device: str,
        savedir: str
    ) -> None:
        """
        联邦学习算法基础类
        :param nn.Module global_net: 全局模型
        :param dict nets: 所有的局部模型
        :param dict datasets: 拆分的所有的数据集
        :param dict test_dataset: 测试数据集
        :param int nk_parties: 每轮选取多少个节点融合
        :param int E: 本地的epoch
        :param int comm_round: 通信的轮数
        :param float lr: 优化器学习率
        :param int batch_size: 优化器的batch大小
        :param float weight_decay: 优化器权重衰减系数
        :param str optim_name: 优化器的名称
        :param str device: 训练设备， GPU或者CPU
        :param str savedir: 模型保存路径
        """
        self._global_net = global_net
        self._nets = nets
        self._datasets = datasets
        self._test_dataset = test_dataset
        self._nk_parties = nk_parties
        self._E = E
        self._comm_round = comm_round
        
        self._lr = lr
        self._bs = batch_size
        self._weight_decay = weight_decay
        self._optim_name = optim_name

        self._device = torch.device(f"cuda") if device == "cuda" else torch.device("cpu")

        self._savedir = savedir
        
    def _valid(self, net, dataset, bs, device):
        net = net.to(device)
        test_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=False, drop_last=True)
        with torch.no_grad():
            net.eval()
            correct, total = 0, 0
            for _, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)
                pred_y = net(x)
                total += y.shape[0]
                _, pred_label = torch.max(pred_y.data, 1)
                correct += (pred_label == y.data).sum().item()
            net = net.to(torch.device("cpu"))
            return correct / total

    def _optimizer(self, optim_name, net, lr, weight_decay: float = 1e-5):
        if optim_name == "sgd":
            return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optim_name == "adam":
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay)
        elif optim_name == "amsgrad":
            return optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            return optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)

    def _sample_nets(self, nets, nk_parties):
        if nk_parties > len(nets):
            logging.info(f"Error, can not sample {nk_parties} client over {len(nets)} clients")
            return nets
        else:
            idxs = [idx for idx in range(len(nets))]
            sampled = np.random.choice(idxs, size=nk_parties, replace=False)
            samples = {}
            for idx in sampled:
                samples[idx] = nets[idx]
            samples = dict(sorted(samples.items(), key=operator.itemgetter(0)))
            return samples

    def start(self):
        pass

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, device: torch.device):
        pass

    def _aggregate(self, net_w_lst: list, ratios: list):
        pass
