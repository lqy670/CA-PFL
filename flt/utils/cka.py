#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import math
import numpy as np
from typing import List, Optional


__all__ = ["get_cka_pairs", "get_cka_matrix", "centered_kernel_alignment"]


class Kernel:
    @staticmethod
    def linear(x: np.ndarray, y: np.ndarray):
        """
        线型函数核
        :param x:
        :param y:
        :return:
        """
        return x @ y.T

    @staticmethod
    def rbf(x: np.ndarray, y: np.ndarray, sigma: Optional[float] = None):
        """
        rbf kernel, 径向基函数核
        :param x: 矩阵一
        :param y: 矩阵二
        :param sigma: 径向基参数
        :return:
        """
        # 计算两个矩阵之间的gram的矩阵
        gram_x = x @ y.T
        # 对角线上元素设置为 0
        diag_gx = np.diag(gram_x) - gram_x # type: ignore
        # 矩阵与其转置相加, 即生成对称矩阵
        k_x = diag_gx + diag_gx.T
        if sigma is None:
            # 中位数
            median_dist = np.median(k_x[k_x != 0])
            sigma = math.sqrt(median_dist)
        k_x *= - 0.5 / (sigma * sigma)
        k_x = np.exp(k_x)
        return k_x


def h_s_independence_criterion(k: np.ndarray, l: np.ndarray):
    """
    希尔伯特-施密特独立性准则, 衡量两个变量的一个分布差异
    :param k: 矩阵变量
    :param l: 矩阵变量
    :return:
    """
    n = k.shape[0]
    identity = np.identity(n)
    h = identity - np.ones((n, n)) / n
    return np.trace(k @ h @ l @ h) / (n - 1) ** 2


def get_kernel(kernel_str: str):
    """
    获取CKA使用的核函数
    :param str method: 核函数名称
    :return : 
    """
    if kernel_str is None:
        return Kernel.linear
    func = getattr(Kernel, kernel_str)
    if func is None:
        return Kernel.linear
    else:
        return func


def centered_kernel_alignment(x: np.ndarray, y: np.ndarray, kernel: Optional[str] = None):
    """
    计算cka相似度
    :param x: 需要计算的激活矩阵, [n, p1]
    :param y: 需要计算的激活矩阵, [n, p2]
    :param kernel: 采用的核函数
    :return:
    """
    if kernel is None:
        kernels = get_kernel("linear")
    else:
        kernels = get_kernel(kernel)
    gram_k = kernels(x, x)
    gram_l = kernels(y, y)
    h_s_i_c = h_s_independence_criterion(gram_k, gram_l)
    var_k = np.sqrt(h_s_independence_criterion(gram_k, gram_k))
    var_l = np.sqrt(h_s_independence_criterion(gram_l, gram_l))
    return h_s_i_c / (var_k * var_l)


def get_cka_matrix(acts1: List[np.ndarray], acts2: List[np.ndarray], kernel: str = "linear"):
    """
    两个激活矩阵列表相互进行cka计算, 即 activations_1 与 activations_2每个元素都进行计算
    :param activations_1: 激活矩阵, [n, p1]
    :param activations_2: 激活矩阵, [n, p2]
    :param kernel: 采用核函数
    :return:
    """
    num_layers_1, num_layers_2 = len(acts1), len(acts2)
    cka_matrix = np.zeros((num_layers_1, num_layers_2))
    # 两个列表激活矩阵个数是否相等
    symmetric = (num_layers_1 == num_layers_2)
    for i in range(num_layers_1):
        # 对称矩阵只需要求出一半
        if symmetric:
            for j in range(i, num_layers_2):
                x, y = acts1[i], acts2[j]
                cka_temp = centered_kernel_alignment(x, y, kernel)
                cka_matrix[i, j] = cka_temp
                cka_matrix[j, i] = cka_temp
        else:
            for j in range(num_layers_2):
                x, y = acts1[i], acts2[j]
                cka_temp = centered_kernel_alignment(x, y, kernel)
                cka_matrix[i, j] = cka_temp
    return cka_matrix


def get_cka_pairs(acts1: List[np.ndarray], acts2: List[np.ndarray], kernel: str = "linear"):
    """
    两两之间进行cka的计算, activations_1 与 activations_2 对应的层进行计算
    :param acts1: 激活矩阵, [n, p1]
    :param acts2: 激活矩阵, [n, p2]
    :param kernel: 采用的核函数
    :return:
    """
    cka_pairs = []
    for act1, act2 in zip(acts1, acts2):
        # 样本数据不足
        if act1.shape[0] == act2.shape[0] and act1.shape[0] == 1:
            cka_pairs.append(0.0)
        else:
            cka_pairs.append(centered_kernel_alignment(act1, act2, kernel))
    return cka_pairs


if __name__ == '__main__':
    import torch
    from torch import nn
    # # Samples
    # n = 100
    # # Representation dim model 1
    # p1 = 64
    # # Representation dim model 1
    # p2 = 64
    #
    # # Generate X
    # X = np.random.normal(size=(n, p1))
    # Y = np.random.normal(size=(n, p2))
    #
    # # Center columns
    # X = X - np.mean(X, 0)
    # Y = Y - np.mean(Y, 0)
    #
    # cosine = torch.nn.CosineSimilarity(dim=-1)
    # sim = cosine(torch.from_numpy(X),torch.from_numpy(Y))
    # print(sim)
    #
    # sim = centered_kernel_alignment(X, Y)
    # print(sim)
    #
    # sim = centered_kernel_alignment(X, X)
    # print(sim)
    #
    # sim = centered_kernel_alignment(X, Y, kernel="rbf")
    # print(sim)
    #
    # sim = centered_kernel_alignment(X, X, kernel="rbf")
    # print(sim)

    X = np.random.normal(size=(16, 512, 1, 1)).reshape(16, 512)
    Y = np.random.normal(size=(16, 512, 1, 1)).reshape(16, 512)
    Z = np.random.normal(size=(16, 512, 1, 1)).reshape(16, 512)
    sim1 = centered_kernel_alignment((X), (Y), kernel="rbf")
    sim2 = centered_kernel_alignment((X), (Z), kernel="rbf")
    logits = sim1.reshape(-1, 1)
    logits = torch.cat([torch.from_numpy(logits), torch.from_numpy(sim2.reshape(-1, 1))], dim=1)
    logits /= 0.5
    labels = torch.zeros(1).cuda().long()
    criterion = nn.CrossEntropyLoss().to("cuda:0")
    loss2 = criterion(logits.to("cuda:0"), labels.to("cuda:0"))
    print(loss2)
    # print(sim)
    # sim = cosine(torch.from_numpy(X), torch.from_numpy(Y))
    # print(sim)

