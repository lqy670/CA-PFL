#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import copy
import torch
import logging
from torch import nn
from torch.utils import data
from flt.algorithms.base import FedBase


class MOON(FedBase):
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
            mu: float,
            temperature: float,
            pool_size: int,
            *args, **kwargs
    ) -> None:
        """
        MOON 算法
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
        :param str device: 训练设备， GPU或者CPU
        :param str savedir: 模型保存路径
        :param str mu: 对比损失的权重
        :param str temperature: 对比损失计算权重
        :param str pool_size: 采用多大的旧模型池
        """
        super().__init__(
            global_net, nets, datasets, test_dataset, nk_parties, E, comm_round,
            lr, batch_size, weight_decay, optim_name, device, savedir)
        # self._nets = {k: wrapper_net(net) for k, net in nets.items()}
        self._global_test_dataset = global_test_dataset
        self._nets = {k: net for k, net in nets.items()}

        # MOON 算法相关，对比损失权重，对比相似度温度系数，使用多少个旧模型计算对比损失
        self._mu = mu
        self._temperature = temperature
        self._pool_size = pool_size

        self._args = args
        self._kwargs = kwargs

        # 设置全局模型不可训练
        self._global_net = self._require_grad_false(self._global_net)
        self._prev_nets = {}
        for key, net in self._nets.items():
            self._prev_nets[key] = [self._require_grad_false(copy.deepcopy(net))]

    def start(self):
        # 遍历所有的通信轮数，训练模型，融合模型
        for round in range(self._comm_round):
            global_w = self._global_net.state_dict()
            logging.info(f"[Round] {round + 1} / {self._comm_round} start")
            # 选择部分或者全部节点进行训练
            samples = self._sample_nets(self._nets, self._nk_parties)
            net_w_lst, ratios = [], []
            step1_avg_acc = 0.0
            step2_avg_acc = 0.0
            for idx, (key, net) in enumerate(samples.items()):
                logging.info(f"  >>> [Local Train] client: {key} / [{idx + 1}/{len(samples)}]")
                net.load_state_dict(global_w)

                step1_acc = self._valid(net, self._test_dataset[key], self._bs, self._device)
                step1_avg_acc += step1_acc

                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=self._weight_decay)

                # 选取先前的模型
                prev_nets_pool = self._prev_nets[key] if round != 0 else []
                # 调用测试全局模型
                self._global_net = self._require_grad_false(self._global_net)

                net, last_acc = self._train(net, dataset=self._datasets[key], test_dataset=self._test_dataset[key],
                                            optimizer=optimizer, bs=self._bs, E=self._E, old_nets=prev_nets_pool,
                                            mu=self._mu, temperature=self._temperature, device=self._device)
                # net_w_lst.append(copy.deepcopy(net.state_dict()))
                net_w_lst.append(net.state_dict())
                ratios.append(len(self._datasets[key]))
                # 保存所有的旧模型
                old_nets = self._prev_nets[key]
                if len(old_nets) < self._pool_size:
                    old_nets.append(self._require_grad_false(copy.deepcopy(net)))
                    self._prev_nets[key] = old_nets
                else:
                    for _ in range(len(old_nets) + 1 - self._pool_size):
                        del old_nets[0]
                    old_nets.append(self._require_grad_false(copy.deepcopy(net)))
                    self._prev_nets[key] = old_nets
                step2_avg_acc += last_acc
            # TODO step1.每轮联邦学习开始时 使用全局模型更新本地模型后 本地模型在各个客户端的测试集上的准确率
            logging.info(
                f"[Gloabl] Round: {round + 1}, Client model before training - Clients Average Acc: {step1_avg_acc / self._nk_parties}")
            # TODO step2.融合全局模型前所有本地模型在各个测试集上的的平均准确率
            logging.info(
                f"[Gloabl] Round: {round + 1}, Client model after training - Clients Average Acc: {step2_avg_acc / self._nk_parties}")
            # 模型聚合
            global_w = self._aggregate(net_w_lst, ratios)
            self._global_net.load_state_dict(global_w)
            # TODO step3.融合后的全局模型在各个测试集上的平均准确率
            step3_avg_acc = 0.0
            for idx, (key, net) in enumerate(samples.items()):
                acc = self._valid(self._global_net, self._test_dataset[key], self._bs, self._device)
                step3_avg_acc += acc
            # acc = self._valid(self._global_net, self._global_test_dataset, self._bs, self._device)
            logging.info(f"[Gloabl] Round: {round + 1}, Global model - Clients Average Acc: {step3_avg_acc / self._nk_parties}")
            # 保存最后五个模型
            if round >= 45:
                if not os.path.exists(f"{self._savedir}/models/"):
                    os.makedirs(f"{self._savedir}/models/")
                torch.save(
                    self._global_net.state_dict(), f"{self._savedir}/models/global_round_{round + 1}.pth"
                )

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, old_nets: list, mu: float,
               temperature: float, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        cosine = torch.nn.CosineSimilarity(dim=-1)
        last_acc = 0.0
        for epoch in range(E):
            epoch_loss_lst = []
            epoch_loss1_lst = []
            epoch_loss2_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                _, prob, pred = net(x)
                loss1 = criterion(pred, y)

                # 计算全局模型的输出
                self._global_net.to(device)
                _, global_prob, _ = self._global_net(x)
                # self._global_net.to("cpu")

                positive = cosine(prob, global_prob)
                logits = positive.reshape(-1, 1)

                # 计算每一个旧模型的输出
                for old_net in old_nets:
                    old_net = old_net.to(device)
                    _, old_prob, _ = old_net(x)
                    negative = cosine(prob, old_prob)
                    logits = torch.cat([logits, negative.reshape(-1, 1)], dim=1)
                    # old_net.to("cpu")

                logits /= temperature
                labels = torch.zeros(x.size(0)).cuda().long()
                loss2 = mu * criterion(logits, labels)
                loss = loss1 + loss2

                loss.backward()
                optimizer.step()

                epoch_loss1_lst.append(loss1.item())
                epoch_loss2_lst.append(loss2.item())
                epoch_loss_lst.append(loss.item())
            epoch_loss1_lst = [0] if len(epoch_loss1_lst) == 0 else epoch_loss1_lst
            epoch_loss2_lst = [0] if len(epoch_loss2_lst) == 0 else epoch_loss2_lst
            epoch_loss_lst = [0] if len(epoch_loss_lst) == 0 else epoch_loss_lst
            logging.info(
                f"    >>> [Local Train] Epoch: {epoch + 1}, "
                f"Optim Loss: {(sum(epoch_loss1_lst) / len(epoch_loss1_lst)):.6f}, "
                f"Contrast Loss: {(sum(epoch_loss2_lst) / len(epoch_loss2_lst)):.6f}, "
                f"Total Loss: {(sum(epoch_loss_lst) / len(epoch_loss_lst)):.6f}"
            )

            with torch.no_grad():
                net.eval()
                correct, total = 0, 0
                for _, (x, y) in enumerate(test_dataloader):
                    x, y = x.to(device), y.to(device)
                    _, _, pred_y = net(x)
                    total += y.shape[0]
                    _, pred_label = torch.max(pred_y.data, 1)
                    correct += (pred_label == y.data).sum().item()
            logging.info(f"    >>> [Local Train] Epoch: {epoch + 1}, Acc: {(correct / total):.6f}")
            last_acc = correct / total
        net = net.to(torch.device("cpu"))
        return net, last_acc

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

    def _valid(self, net, dataset, bs, device):
        net = net.to(device)
        test_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=False, drop_last=True)
        with torch.no_grad():
            net.eval()
            correct, total = 0, 0
            for _, (x, y) in enumerate(test_dataloader):
                x, y = x.to(device), y.to(device)
                _, _, pred_y = net(x)
                total += y.shape[0]
                _, pred_label = torch.max(pred_y.data, 1)
                correct += (pred_label == y.data).sum().item()
            net = net.to(torch.device("cpu"))
            return correct / total

    def _require_grad_false(self, net):
        net.eval()
        # 设置全局模型不可训练
        for param in net.parameters():
            param.requires_grad = False
        return net
