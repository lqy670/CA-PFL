#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import copy
import torch
import logging
from torch import nn
from torch.utils import data
from flt.algorithms.base import FedBase
import numpy as np
from sklearn.cluster import AgglomerativeClustering


class Ditto(FedBase):
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
        FedAvg 算法
        :param nn.Module global_net: 全局模型
        :param dict nets: 所有的局部模型
        :param dict datasets: 拆分的所有的训练集
        :param dict test_dataset: 测试数据集
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
        self._global_test_dataset = global_test_dataset
        self._args = args
        self._kwargs = kwargs

        # 社区模型模板
        self._community_net = copy.deepcopy(self._global_net)
        # 各社区模型字典
        self._cluster_models = {}
        self._clients_W = {}  # 缓存客户端模型参数
        self._clients_dW = {}  # 缓存客户端模型参数变化量
        self._clients_W_old = {}  # 缓存客户端模型训练的初始参数
        for key, net in self._nets.items():
            self._clients_W[key] = {key1: value for key1, value in net.named_parameters()}
            self._clients_dW[key] = {key1: torch.zeros_like(value) for key1, value in net.named_parameters()}
            self._clients_W_old[key] = {key1: torch.zeros_like(value) for key1, value in net.named_parameters()}
        self._EPS_1 = 0.4
        self._EPS_2 = 1.6
        self._lam = 0.1

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss()
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
                f"    >>> [Local Train] Epoch: {epoch + 1}, Loss: {sum(epoch_loss_lst) / len(epoch_loss_lst)}, Acc: {correct / total}")
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

    def start(self):
        # 联邦社区初始化为[[0,1,2,3,4,...,9]],即认为都在一个社区
        cluster_indices = [[client for client in range(self._nk_parties)]]
        # 初始化0号社区模型，这里设置不可训练没有影响
        self._cluster_models[0] = copy.deepcopy(self._global_net)

        for round in range(self._comm_round):
            logging.info(f"[Round] {round + 1} / {self._comm_round} start")
            logging.info(f">>> [Round] {round + 1} -- [Community] {cluster_indices}")
            # 选择部分或者全部节点进行训练
            samples = self._sample_nets(self._nets, self._nk_parties)
            net_w_lst, ratios = [], []
            step1_avg_acc = 0.0
            step2_avg_acc = 0.0
            for idx, (key, net) in enumerate(samples.items()):
                logging.info(f"  >>> [Local Train] client: {key} / [{idx + 1}/{len(samples)}]")
                # 获取当前客户端所在社区的社区模型特征提取部分
                community_net = self._cluster_models[self.find_cluster(cluster_indices, key)]
                community_net_w = community_net.state_dict()
                net.load_state_dict(community_net_w)
                self._clients_W[key] = {key1: value for key1, value in net.named_parameters()}  # 更新该客户端参数记录

                step1_acc = self._valid(net, self._test_dataset[key], self._bs, self._device)
                step1_avg_acc += step1_acc
                # 记录训练前客户端模型权重
                self.copy(target=self._clients_W_old[key], source=self._clients_W[key])

                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=self._weight_decay)
                net, last_acc = self._train(
                    net, dataset=self._datasets[key], test_dataset=self._test_dataset[key],
                    optimizer=optimizer, bs=self._bs, E=self._E, device=self._device
                )

                self._clients_W[key] = {key1: value for key1, value in net.named_parameters()}  # 更新该客户端参数记录
                # 更新各客户端模型权重差值
                self.subtract_(target=self._clients_dW[key], minuend=self._clients_W[key],
                               subtrahend=self._clients_W_old[key])

                net_w_lst.append(net.state_dict())
                ratios.append(len(self._datasets[key]))
                step2_avg_acc += last_acc
            # TODO step1.每轮联邦学习开始时 使用全局模型更新本地模型后 本地模型在各个客户端的测试集上的准确率
            logging.info(
                f"[Gloabl] Round: {round + 1}, Client model before training - Clients Average Acc: {step1_avg_acc / self._nk_parties}")
            # TODO step2.融合全局模型前所有本地模型在各个测试集上的的平均准确率
            logging.info(
                f"[Gloabl] Round: {round + 1}, Client model after training - Clients Average Acc: {step2_avg_acc / self._nk_parties}")
            # 计算所有clients的参数相似度的邻接矩阵
            similarities = self.compute_pairwise_similarities()
            cluster_indices_new = []
            for idc in cluster_indices:
                max_norm = self.compute_max_update_norm([i for i in idc])
                mean_norm = self.compute_mean_update_norm([i for i in idc])
                if mean_norm < self._EPS_1 and max_norm > self._EPS_2 and len(idc) > 2:
                    c1, c2 = self.cluster_clients(similarities[idc][:, idc])
                    cluster_indices_new += [list(np.array(idc)[c1]),list(np.array(idc)[c2])]
                else:
                    cluster_indices_new += [idc]
            if len(cluster_indices_new) > len(cluster_indices):
                logging.info(f"    >>> [Round] {round + 1} -- [Community] {cluster_indices_new}")
            cluster_indices = cluster_indices_new
            # 按照社区进行聚合
            self._cluster_models = self.cluster_aggregate(cluster_indices, net_w_lst, ratios)

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

    def _require_grad_false(self, net):
        net.eval()
        # 设置全局模型不可训练
        for param in net.parameters():
            param.requires_grad = False
        return net

    # 以下为社区检测相关
    # 逐(i,j)对参数计算相似度 这里的sources为所有client参数变化量dW组成的列表
    def pariwise_angles(self, sources):
        angles = torch.zeros(len(sources), len(sources))
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                s1 = self.flatten(source1)
                s2 = self.flatten(source2)
                angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)
        return angles.numpy()

    def compute_pairwise_similarities(self):
        return self.pariwise_angles([value for _, value in self._clients_dW.items()])

    def flatten(self, source):
        return torch.cat([value.flatten() for value in source.values()])

    def cache_cluster(self, cluster_idcs, c_round):
        self._cluster_cache.append(cluster_idcs)
        self._round_cache.append(c_round)

    def get_weighted_edges(self, similarities, idc):
        weighted_edges = []
        for i, node_id1 in zip(range(similarities.shape[0]), idc):
            for j, node_id2 in zip(range(similarities.shape[1]), idc):
                # 不包括自环和负权
                if i != j:
                    weighted_edges.append((node_id1, node_id2, max(1e-12, similarities[i][j])))
        return weighted_edges

    def copy(self, target, source):
        for name in target:
            target[name].data = source[name].data.clone()

    def subtract_(self, target, minuend, subtrahend):
        for name in target:
            target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

    # 每个社区聚合模型 这里是社区id与其对应的模型权重{id:weight}
    def cluster_aggregate(self, client_clusters: list, net_w_lst: list, ratios: list):
        cluster_models = {}
        for id, client in enumerate(client_clusters):  # 每个簇
            sample_num = 0.0
            for i in client:
                sample_num += ratios[i]
            cluster_model = copy.deepcopy(net_w_lst[client[0]])
            for key in cluster_model.keys():
                if "num_batches_tracked" not in key:
                    cluster_model[key] *= (ratios[client[0]] / sample_num)
            for key in cluster_model.keys():
                for i in client[1:]:
                    if "num_batches_tracked" not in key:
                        cluster_model[key] += (ratios[i] / sample_num) * net_w_lst[i][key]
            net = copy.deepcopy(self._community_net)
            net.load_state_dict(cluster_model)
            net = self._require_grad_false(net)
            cluster_models[id] = net
        return cluster_models

    # 根据客户端id找到自己的社区编号
    def find_cluster(self, cluster_indices, key):
        for i, cluster in enumerate(cluster_indices):
            if key in cluster:
                return i

    def compute_max_update_norm(self, cluster):
        return np.max([torch.norm(self.flatten(self._clients_dW[client])).item() for client in cluster])

    def compute_mean_update_norm(self, cluster):
        return torch.norm(
            torch.mean(torch.stack([self.flatten(self._clients_dW[client]) for client in cluster]), dim=0)).item()

    def cluster_clients(self, S):
        clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

        c1 = np.argwhere(clustering.labels_ == 0).flatten()
        c2 = np.argwhere(clustering.labels_ == 1).flatten()
        return c1, c2
