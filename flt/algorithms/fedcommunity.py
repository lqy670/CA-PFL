#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import copy
import random

import torch
import logging
from torch import nn
from torch.utils import data
from flt.algorithms.base import FedBase
from flt.graph_cluster.community_louvain import best_partition, modularity
from flt.utils.net_ops import NetOps
import igraph as ig
import leidenalg
import networkx as nx


class FedCommunity(FedBase):
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
            savedir: str,
            mu: float,
            temperature: float,
            pool_size: int,
            p,
            split,
            end,
            *args, **kwargs
    ) -> None:
        """
        FedCC 算法
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

        # 加载 global net 的神经网络操作 TODO 由于不仅仅需要加载global net 还需要加载community net 因此在后面再定义
        # self._nn_ops = NetOps(global_net)
        # fe = self._nn_ops.get_fe_param(global_net.state_dict())
        # logging.info(f"[feature extractor]: {list(fe.keys())}")
        # cla = self._nn_ops.get_cla_param(global_net.state_dict())
        # logging.info(f"[classifier]: {list(cla.keys())}")

        # self._nets = {k: wrapper_net(net) for k, net in nets.items()}
        # self._global_test_dataset = global_test_dataset
        self._nets = {k: net for k, net in nets.items()}

        # MOON 算法相关，对比损失权重，对比相似度温度系数，使用多少个旧模型计算对比损失
        self._mu = mu
        self._temperature = temperature
        self._pool_size = pool_size
        self._p = p
        self._split = split
        self._end = end

        self._args = args
        self._kwargs = kwargs

        # self._prev_nets = {}
        # for key, net in self._nets.items():
        #     self._prev_nets[key] = [self._require_grad_false(copy.deepcopy(net))]

        # community detection init
        # 社区模型模板
        self._community_net = copy.deepcopy(self._global_net)
        # 设置全局模型不可训练
        self._global_net = self._require_grad_false(self._global_net)
        # 各社区模型字典
        self._cluster_models = {}
        self._clients_W = {}  # 缓存客户端模型参数
        self._clients_dW = {}  # 缓存客户端模型参数变化量
        self._clients_W_old = {}  # 缓存客户端模型训练的初始参数
        for key, net in self._nets.items():
            self._clients_W[key] = {key1: value for key1, value in net.named_parameters()}
            self._clients_dW[key] = {key1: torch.zeros_like(value) for key1, value in net.named_parameters()}
            self._clients_W_old[key] = {key1: torch.zeros_like(value) for key1, value in net.named_parameters()}
        self._pre_mod = 0.2  # 前轮的模块度,这里初始化为0.2
        self._min_val = 0  # 模块度增幅

    def start(self):
        # 联邦社区初始化为[[0,1,2,3,4,...,9]],即认为都在一个社区
        cluster_indices = [[client for client in range(self._nk_parties)]]
        # 初始化0号社区模型，这里设置不可训练没有影响
        self._cluster_models[0] = self._require_grad_false(copy.deepcopy(self._global_net))
        # 初始化图,按照初始的client参数计算相似度,初始时client参数都为0,图的权重都为0
        similarities = self.compute_pairwise_similarities()  # 10*10的矩阵
        graph = nx.Graph()  # 使用network初始化图
        # 这里add_weighted_edges_from方法要求传入[(0,1,0.3),(1,2,0.5)]这种元组列表结构
        graph.add_weighted_edges_from(self.get_weighted_edges(similarities, range(self._nk_parties)))
        # 遍历所有的通信轮数，训练模型，融合模型
        for round in range(self._comm_round):
            # global_w = self._global_net.state_dict()
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
                nn_ops = NetOps(community_net)
                fe = nn_ops.get_fe_param(community_net_w)
                # 获取当前客户端的本地模型参数
                local_w = net.state_dict()
                # 组合社区模型特征提取部分与本地模型的个性化层 相同的键做了更新
                local_w.update(fe)
                net.load_state_dict(local_w)
                self._clients_W[key] = {key1: value for key1, value in net.named_parameters()}  # 更新该客户端参数记录
                step1_acc = self._valid(net, self._test_dataset[key], self._bs, self._device)
                step1_avg_acc += step1_acc
                # 记录训练前客户端模型权重
                self.copy(target=self._clients_W_old[key], source=self._clients_W[key])
                optimizer = self._optimizer(self._optim_name, net, lr=self._lr, weight_decay=self._weight_decay)

                # 调用测试全局模型
                self._global_net = self._require_grad_false(self._global_net)
                # community_net = self._require_grad_false(community_net)
                # 调用所有社区模型 当存在两个及以上个社区模型时
                net, last_acc = self._train(net, dataset=self._datasets[key], test_dataset=self._test_dataset[key],
                                            optimizer=optimizer, bs=self._bs, E=self._E,
                                            mu=self._mu, temperature=self._temperature, device=self._device)
                # net_w_lst.append(copy.deepcopy(net.state_dict()))
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
            logging.info(similarities)
            # print(similarities)
            # 更新图的相似度矩阵
            graph.add_weighted_edges_from(self.get_weighted_edges(similarities, range(self._nk_parties)))
            cluster_indices_new = []
            # 使用leiden方法计算社区,这里先把networkx转为了igraph
            # g = ig.Graph.from_networkx(graph)
            # part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)  # 社区划分结果
            # cur_mod = part.modularity
            par_idcs, par_dict = self.community_detection_method(graph)
            cur_mod = modularity(par_dict, graph)
            logging.info(f"the community modularity is {cur_mod}")
            if len(cluster_indices) == 1:
                if cur_mod > 0.4:
                    cluster_indices_new += par_idcs
                    self._pre_mod = cur_mod
                else:
                    cluster_indices_new = cluster_indices
            else:
                if cur_mod > 0.2:
                    cluster_indices_new += par_idcs
                    self._pre_mod = cur_mod
                else:
                    cluster_indices_new = cluster_indices

            # if cur_mod - self._pre_mod > self._min_val:
            #     # for x in part:
            #     #     cluster_indices_new += x
            #     cluster_indices_new += par_idcs
            #     self._pre_mod = cur_mod
            # else:
            #     cluster_indices_new = cluster_indices

            cluster_indices = cluster_indices_new
            # 全局模型聚合
            global_w = self._aggregate(net_w_lst, ratios)
            self._global_net.load_state_dict(global_w)
            # 按照社区进行聚合
            self._cluster_models = self.cluster_aggregate(cluster_indices, net_w_lst, ratios)
            # TODO 为了融合全局知识，对各社区模型进行全局EMA融合，融合权重暂时根据轮数确定，若只有一个社区则不进行
            # TODO 实现基于表征相似度的EMA融合方法 u=min(sim(rep1,rep2),1)
            # if len(cluster_indices) != 1:
            #     logging.info(cluster_indices_new)
            #     logging.info(f"the ema p is {self._p}")
            #     # 动量更新各社区模型 这里zip的元素个数与最短的一致
            #     self._global_net = self._global_net.to(self._device)
            #     # l = (((round + 1) / self._comm_round) + 1) / 2
            #     for key, value in self._cluster_models.items():
            #         value = value.to(self._device)
            #         # 随机选一个客户端的数据集，用全局模型与当前社区模型计算随机一个minibatch的表征，计算CKA相似度
            #         # cli = random.choice(cluster_indices[key])
            #         # sim = batch_cka(self._datasets[cli],value,self._global_net)
            #         # t = min(,1)
            #         for param_q, param_k in zip(self._global_net.parameters(), value.parameters()):
            #             # param_k.data = param_k.data * self._p + param_q * (1-self._p)
            #             param_k.data = param_k.data * self._p + param_q * (1-self._p)

            # 保存最后五个模型
            # if round >= 45:
            #     if not os.path.exists(f"{self._savedir}/models/"):
            #         os.makedirs(f"{self._savedir}/models/")
            #     torch.save(
            #         self._global_net.state_dict(), f"{self._savedir}/models/global_round_{round + 1}.pth"
            #     )

    def _train(self, net, dataset, test_dataset, optimizer, bs: int, E: int, mu: float,
               temperature: float, device: torch.device):
        train_dataloader = data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True, drop_last=True)
        test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)

        net = net.to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        cosine = torch.nn.CosineSimilarity(dim=-1)
        last_acc = 0.0
        for epoch in range(E):
            epoch_loss_lst = []
            net.train()
            for _, (x, y) in enumerate(train_dataloader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                _, prob, pred = net(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss_lst.append(loss.item())

            epoch_loss_lst = [0.0] if len(epoch_loss_lst) == 0 else epoch_loss_lst
            logging.info(
                f"    >>> [Local Train] Epoch: {epoch + 1}, "
                f"Loss: {(sum(epoch_loss_lst) / len(epoch_loss_lst)):.6f}"
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

    # 以下为社区检测相关
    # 逐(i,j)对参数计算相似度 这里的sources为所有client参数变化量dW组成的列表
    def pariwise_angles(self, sources):
        logging.info(f"{self._split}-{self._end}")
        angles = torch.zeros(len(sources), len(sources))
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources):
                s1 = self.flatten(dict_slice(source1, self._split, self._end))
                s2 = self.flatten(dict_slice(source2, self._split, self._end))
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

    def community_detection_method(self, graph):
        par_dict = best_partition(graph, resolution=graph.number_of_nodes())
        # 反转kv
        cluster_to_clients = {}
        for client_id, cluster_id in par_dict.items():
            if cluster_id not in cluster_to_clients.keys():
                cluster_to_clients[cluster_id] = [client_id]
            else:
                cluster_to_clients[cluster_id].append(client_id)
        cluster_indices_new = []
        for c_clitnts in cluster_to_clients.values():
            cluster_indices_new.append(c_clitnts)
        return cluster_indices_new, par_dict


def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice
