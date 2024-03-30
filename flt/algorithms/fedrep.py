#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import copy
import torch
import logging
from torch import nn
from torch.utils import data
from flt.utils.net_ops import NetOps
from flt.algorithms.base import FedBase


class FedRep(FedBase):
    def __init__(
            self,
            global_net,
            nets: dict,
            datasets: dict,
            test_dataset: dict,
            global_test_dataset: data.Dataset,
            nk_parties: int,
            E: int,
            comm_round: int,
            lr: float,
            batch_size: int,
            weight_decay: float,
            optim_name: str,
            device: str,
            savedir: str,
            *args, **kwargs
    ) -> None:
        """
        FedPer 个性化联邦学习算法 Federated Learning with Personalization Layers (https://arxiv.org/abs/1912.00818v1)
        :param nn.Module global_net: 全局模型
        :param dict nets: 所有的局部模型
        :param dict datasets: 拆分的所有的数据集
        :param dcit test_dataset: 测试数据集
        :param data.Dataset global_test_dataset: 全局测试数据集
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
        super().__init__(
            global_net, nets, datasets, test_dataset, nk_parties, E, comm_round,
            lr, batch_size, weight_decay, optim_name, device, savedir)

        # 加载 global net 的神经网络操作
        self._nn_ops = NetOps(global_net)

        # for idx, (name, module) in enumerate(self._nn_ops.ntlst):
        #     logging.info(f"{idx:^3d} -> {name:<35s} -> {module}")
        # feature extractor
        fe = self._nn_ops.get_fe_param(global_net.state_dict())
        logging.info(f"[feature extractor]: {list(fe.keys())}")
        cla = self._nn_ops.get_cla_param(global_net.state_dict())
        logging.info(f"[classifier]: {list(cla.keys())}")

        self._global_test_dataset = global_test_dataset
        self._args = args
        self._kwargs = kwargs

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss()

        # 先更新10步head层
        # 冻结base层
        for name, param in net.named_parameters():
            if "block_1" in name or "block_2" in name or "block_3" in name or "block_4" in name or "block_5" in name or "block_6" in name:
                param.requires_grad = False
            if "fc" in name:
                param.requires_grad = True

        head_last_acc = 0.0
        for epoch in range(10):
            epoch_loss_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred_y = net(x)
                loss = criterion(pred_y, y)
                loss.backward()
                optimizer.step()
                epoch_loss_lst.append(loss.item())
            epoch_loss_lst = [0.0] if len(epoch_loss_lst) == 0 else epoch_loss_lst

            with torch.no_grad():
                net.eval()
                correct, total = 0, 0
                for _, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(device), y.to(device)
                    pred_y = net(x)
                    total += y.shape[0]
                    _, pred_label = torch.max(pred_y.data, 1)
                    correct += (pred_label == y.data).sum().item()
            logging.info(
                f"    >>> [Local Head Train] Epoch: {epoch + 1}, Loss: {sum(epoch_loss_lst) / len(epoch_loss_lst)}, Acc: {correct / total}")
            head_last_acc = correct / total
        logging.info("------------------------------")
        # 再更新E步base层
        # 冻结head层
        for name, param in net.named_parameters():
            if "block_1" in name or "block_2" in name or "block_3" in name or "block_4" in name or "block_5" in name or "block_6" in name:
                param.requires_grad = True
            if "fc" in name:
                param.requires_grad = False

        last_acc = 0.0
        for epoch in range(E):
            epoch_loss_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred_y = net(x)
                loss = criterion(pred_y, y)
                loss.backward()
                optimizer.step()
                epoch_loss_lst.append(loss.item())
            epoch_loss_lst = [0.0] if len(epoch_loss_lst) == 0 else epoch_loss_lst

            with torch.no_grad():
                net.eval()
                correct, total = 0, 0
                for _, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(device), y.to(device)
                    pred_y = net(x)
                    total += y.shape[0]
                    _, pred_label = torch.max(pred_y.data, 1)
                    correct += (pred_label == y.data).sum().item()
            logging.info(
                f"    >>> [Local Base Train] Epoch: {epoch + 1}, Loss: {sum(epoch_loss_lst) / len(epoch_loss_lst)}, Acc: {correct / total}")
            last_acc = correct / total

        net = net.to(torch.device("cpu"))

        return net, head_last_acc, last_acc

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

    def _concat(self, fe, local_w: dict):
        local_w.update(fe)

    def start(self):
        # 全局模型下发
        for round in range(self._comm_round):
            logging.info(f"[Round] {round + 1} / {self._comm_round} start")
            global_w = self._global_net.state_dict()
            # 选择部分或者全部节点进行训练
            samples = self._sample_nets(self._nets, self._nk_parties)
            net_w_lst, ratios = [], []
            step1_avg_acc = 0.0
            step2_avg_head_acc = 0.0
            step2_avg_base_acc = 0.0
            for idx, (key, net) in enumerate(samples.items()):
                # 获取当前全局模型的特征提取部分
                fe = self._nn_ops.get_fe_param(global_w)
                # 获取当前客户端的本地模型参数
                local_w = net.state_dict()
                # 组合全局模型特征提取部分与本地模型的个性化层 相同的键做了更新
                local_w.update(fe)
                net.load_state_dict(local_w)

                step1_acc = self._valid(net, self._test_dataset[key], self._bs, self._device)
                step1_avg_acc += step1_acc

                logging.info(f"  >>> [Local Train] client: {key} / [{idx + 1}/{len(samples)}]")
                # net.load_state_dict(global_w)
                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=self._weight_decay)
                net, head_last_acc, last_acc = self._train(
                    net, dataset=self._datasets[key], test_dataset=self._test_dataset[key],
                    optimizer=optimizer, bs=self._bs, E=self._E, device=self._device
                )
                net_w_lst.append(net.state_dict())
                ratios.append(len(self._datasets[key]))
                step2_avg_head_acc += head_last_acc
                step2_avg_base_acc += last_acc

            # TODO step1.每轮联邦学习开始时 使用全局模型更新本地模型后 本地模型在各个客户端的测试集上的准确率
            logging.info(
                f"[Gloabl] Round: {round + 1}, Client model before training - Clients Average Acc: {step1_avg_acc / self._nk_parties}")
            # TODO step2.融合全局模型前所有本地模型在各个测试集上的的平均准确率
            logging.info(
                f"[Gloabl] Round: {round + 1}, Client model after training - Clients Average Acc: {step2_avg_base_acc / self._nk_parties}")
            # 模型聚合
            global_w = self._aggregate(net_w_lst, ratios)
            self._global_net.load_state_dict(global_w)
            # 保存最后五个模型
            # if round >= 45:
            #     if not os.path.exists(f"{self._savedir}/models/"):
            #         os.makedirs(f"{self._savedir}/models/")
            #     torch.save(
            #         self._global_net.state_dict(), f"{self._savedir}/models/global_round_{round + 1}.pth"
            #     )
