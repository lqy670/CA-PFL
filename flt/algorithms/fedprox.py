#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import copy
import torch
import logging
from torch import nn
from torch.utils import data
from flt.algorithms.base import FedBase


class FedProx(FedBase):
    def __init__(
        self,
        global_net,
        nets: dict,
        datasets: dict,
        test_dataset: data.Dataset,
        nk_parties: int,
        E: int,
        comm_round: int,
        lr: float,
        batch_size: int,
        weight_decay: float,
        optim_name: str,
        mu: float,
        device: str,
        savedir: str,
        *args, **kwargs
    ) -> None:
        """
        FedProx 算法
        :param nn.Module global_net: 全局模型
        :param dict nets: 所有的局部模型
        :param dict datasets: 拆分的所有的数据集
        :param data.Dataset test_dataset: 测试数据集
        :param int nk_parties: 每轮选取多少个节点融合
        :param int E: 本地的epoch
        :param int comm_round: 通信的轮数
        :param float lr: 优化器学习率
        :param int batch_size: 优化器的batch大小
        :param float weight_decay: 优化器权重衰减系数
        :param str optim_name: 优化器的名称
        :param float mu: FedProx 约束参数
        :param str device: 训练设备， GPU或者CPU
        :param str savedir: 模型保存路径
        """
        super().__init__(
            global_net, nets, datasets, test_dataset, nk_parties, E, comm_round, 
            lr, batch_size, weight_decay, optim_name, device, savedir)

        self._mu = mu
        self._args = args
        self._kwargs = kwargs

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, global_net, mu: float, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        # fedprox proximal term 修正项 约束参数
        global_w = list(global_net.to(device).parameters())
        # 训练过程
        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(E):
            epoch_loss_lst = []
            epoch_optim_lst = []
            epoch_fedprox_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred_y = net(x)
                loss = criterion(pred_y, y)

                fed_prox_reg = torch.tensor(0.0).to(device)
                for key, param in enumerate(net.parameters()):
                    fed_prox_reg += (param - global_w[key]).norm(2)
                fed_prox_reg = (mu / 2) * fed_prox_reg
                total_loss = loss + fed_prox_reg

                total_loss.backward()
                optimizer.step()
                epoch_optim_lst.append(loss.item())
                epoch_fedprox_lst.append(fed_prox_reg.item())
                epoch_loss_lst.append(total_loss.item())
            
            epoch_optim_lst = [0] if len(epoch_optim_lst) == 0 else epoch_optim_lst
            epoch_fedprox_lst = [0] if len(epoch_fedprox_lst) == 0 else epoch_fedprox_lst
            epoch_loss_lst = [0] if len(epoch_loss_lst) == 0 else epoch_loss_lst
            logging.info(
                f"    >>> [Local Train] Epoch: {epoch + 1}, "
                f"Optim Loss: {(sum(epoch_optim_lst) / len(epoch_optim_lst)):.6f}, "
                f"Regular Loss: {(sum(epoch_fedprox_lst) / len(epoch_fedprox_lst)):.6f}, "
                f"Total Loss: {(sum(epoch_loss_lst) / len(epoch_loss_lst)):.6f}"
            )
            with torch.no_grad():
                net.eval()
                correct, total = 0, 0
                for _, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(device), y.to(device)
                    pred_y = net(x)
                    total += y.shape[0]
                    _, pred_label = torch.max(pred_y.data, 1)
                    correct += (pred_label == y.data).sum().item()
            logging.info(f"    >>> [Local Train] Epoch: {epoch + 1}, Acc: {(correct / total):.6f}")
        net = net.to(torch.device("cpu"))
        return net

    def _aggregate(self, net_w_lst: list, ratios: list):
        sample_num = sum(ratios)
        global_w = copy.deepcopy(net_w_lst[0])
        for key in global_w.keys():
            if "num_batches_tracked" not in key:
                global_w[key] *= (ratios[0] / sample_num)
        for key in global_w.keys():
            for i in range(1, len(net_w_lst)):
                if "num_batches_tracked" not in key:
                    global_w[key] += (ratios[i] / sample_num) * net_w_lst[i][key]        
        return global_w

    def start(self):
        for round in range(self._comm_round):
            global_w = self._global_net.state_dict()
            logging.info(f"[Round] {round + 1} / {self._comm_round} start")
            # 选择部分或者全部节点进行训练
            samples = self._sample_nets(self._nets, self._nk_parties)
            net_w_lst, ratios = [], []
            # 遍历所有的采样节点进行训练
            for idx, (key, net) in enumerate(samples.items()):
                logging.info(f"  >>> [Local Train] client: {key} / [{idx + 1}/{len(samples)}]")
                # 给每个节点加载上一轮的全局模型
                net.load_state_dict(global_w)
                # 生成优化器
                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=self._weight_decay)
                net = self._train(
                    net, dataset=self._datasets[key], test_dataset=self._test_dataset, 
                    optimizer=optimizer, bs=self._bs, E=self._E, 
                    global_net=self._global_net, mu=self._mu, device=self._device
                )
                net_w_lst.append(net.state_dict())
                ratios.append(len(self._datasets[key]))
            # 模型聚合
            global_w = self._aggregate(net_w_lst, ratios)
            self._global_net.load_state_dict(global_w)
            acc = self._valid(self._global_net, self._test_dataset, self._bs, self._device)
            # 保存模型
            logging.info(f"[Gloabl] Round: {round + 1}, Acc: {acc}")
            if not os.path.exists(f"{self._savedir}/models/"):
                os.makedirs(f"{self._savedir}/models/")
            torch.save(
                self._global_net.state_dict(), f"{self._savedir}/models/global_round_{round+1}.pth"
            )
