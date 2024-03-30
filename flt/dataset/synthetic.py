#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torchvision
from PIL import Image
from torch.utils import data
from typing import Optional, List, Callable
from torchvision.datasets import CIFAR10


def softmax(arr: np.ndarray, axis: int, keepdims: bool = True):
    arr -= np.max(arr, axis=axis, keepdims=keepdims)
    return np.exp(arr) / np.sum(np.exp(arr), axis=axis, keepdims=keepdims)


class Synthetic(data.Dataset):
    PROB_CLUSTERS = [1.0]

    def __init__(
            self,
            ns: int,
            num_dim: int,
            num_classes: int,
            min_num_samples: int = 5,
            max_num_samples: int = 1000,
            prob_cluster: list = [0.5, 0.5],
        ) -> None:
        """
        Leaf Synthetic 数据集生成
        :param int ns: 节点数量, 即需要生成多少个节点的数据集
        :param int num_classes: 类别数量
        :param int num_dim: 生成数据的维度
        :param list prob_cluster: 类别中心的可能性, defaults to [0.5, 0.5]
        :param int min_num_samples: 每个节点的最小样本数, defaults to 0
        :param int max_num_samples: 每个节点的最大样本数, defaults to 1000
        """
        super().__init__()
        self._ns = ns
        self._num_dim = num_dim
        self._num_classes = num_classes
        self._prob_cluster = prob_cluster
        self._min_num_samples = min_num_samples
        self._max_num_samples = max_num_samples
        self._num_cluster = len(prob_cluster)
        # 从高斯分布中生成矩阵 Q
        self._Q = np.random.normal(
            loc=0.0, scale=1.0, size=(self._num_dim + 1, self._num_classes, len(self._prob_cluster))
        )
        # 创建对角矩阵
        self._sigma = np.zeros((self._num_dim, self._num_dim))
        for i in range(self._num_dim):
            self._sigma[i, i] = (i + 1) ** (-1.2)
        # 产生每个类别中心的均值
        self._mean = self.__produce_cluster_mean()

    def __produce_cluster_mean(self):
        """
        生成每个类别的均值
        :return list: 类别均值
        """
        cluster_mean_lst = []
        # 对于每一个 cluster 生成其类别的均值
        for _ in range(self._num_cluster):
            loc = np.random.normal(loc=0, scale=1.0, size=None)
            mu = np.random.normal(loc=loc, scale=1.0, size=self._num_cluster)
            cluster_mean_lst.append(mu)
        return cluster_mean_lst

    def generator(self):
        # 生成每个节点的样本
        num_samples = self.__generate_task_samples(self._ns, self._min_num_samples, self._max_num_samples)
        tasks = [self.__generate_task(s) for s in num_samples]
        users, user_data = [], {}
        for idx, t in enumerate(tasks):
            x, y = t["x"], t["y"]
            users.append(str(idx))
            user_data[str(idx)] = {
                "x": x, "y": y
            }
        return { "num_samples": num_samples, "users": users, "user_data": user_data }

    def __generate_task(self, num_samples: int):
        """
        基于样本数量生成当前节点的样本
        :param int num_samples: 样本数量
        """
        # 基于 prob cluster 采样出类别下标
        clusteridx = np.random.choice(range(self._num_cluster), size=None, replace=True, p=self._prob_cluster)
        x = self.__generate_x(num_samples)
        y = self.__generate_y(x, cluster_mean=self._mean[clusteridx])
        x = x[:, 1:]
        return {"x": x, "y": y}

    def __generate_x(self, num_samples: int):
        """
        生成样本 x, 生成步骤如下
        step 1. C_t = N(0, I), v_t ~ N(C_t, I)
        step 2. for all i in [1, n_t], x^i_t ~ N(v_t, sigma)
        :param int num_samples: 当前需要生成的样本数量
        """
        B = np.random.normal(loc=0.0, scale=1.0, size=None)
        loc = np.random.normal(B, scale=1.0, size=self._num_dim)
        # 样本维度加 1，用于记录标签，但实际不需要
        samples = np.ones((num_samples, self._num_dim+1))
        # 基于均值 loc 与协方差生成指定大小的多元正态分布
        samples[:, 1:] = np.random.multivariate_normal(mean=loc, cov=self._sigma, size=num_samples)
        return samples

    def __generate_y(self, x: np.ndarray, cluster_mean: np.ndarray):
        """
        生成样本的 y, 生成步骤如下
        step 1. u_t ~ N(mu_t, I)
        step 2. w_t = Q @ u_t
        step 3. y_t^i = argmax(w_t @ x_t + N(0, 0.1*I))

        :param np.ndarray x: 样本 x 
        :param np.ndarray cluster_mean: 类别均值
        """
        u = np.random.normal(loc=cluster_mean, scale=1.0, size=cluster_mean.shape)
        w = np.matmul(self._Q, u)
        num_samples = x.shape[0]
        prob = softmax(
            np.matmul(x, w) + np.random.normal(loc=0.0, scale=0.1, size=(num_samples, self._num_classes)),
            axis=1
        )
        y = np.argmax(prob, axis=1)
        return y
    
    def __generate_task_samples(self, ns: int, min_ns: int, max_ns: int):
        """
        生成每个节点的样本数量
        :param int ns: 节点数
        :param int min_ns: 节点的最少样本数
        :param int max_ns: 节点的最大样本数
        :return : 每个节点的样本数量列表 
        """
        # 使用对数正态分布，生成节点的样本数量
        n_samples = np.random.lognormal(mean=3, sigma=2, size=ns).astype(int)
        n_samples = [min(n + min_ns, max_ns) for n in n_samples]
        return n_samples
    pass


# TODO
class SyntheticWrapper(data.Dataset):
    DATASET = CIFAR10
    def __init__(
            self, 
            root: str, 
            train: bool = True, 
            dataidxs: Optional[List[int]] = None, 
            download: bool = False, 
            transform: Optional[Callable] = None, 
            target_transform: Optional[Callable] = None
        ) -> None:
        self._root = root
        self._train = train
        self._dataidxs = dataidxs
        self._download = download
        self._transform = transform
        self._target_transform = target_transform
        self._datas, self._targets = self._build_datasets()
    
    @property
    def classes(self):
        return self._classes

    @property
    def cls_num_map(self):
        return {key: np.sum(key == self._targets) for key in np.unique(self._targets)}

    def _build_datasets(self):
        wrappered_dataset = self.DATASET(
            root=self._root, 
            train=self._train, 
            transform=self._transform, 
            target_transform=self._target_transform, 
            download=self._download
        )
        self._classes = wrappered_dataset.classes
        if torchvision.__version__ == '0.2.1':
            if self._train:
                datas, targets = wrappered_dataset.train_data, np.array(wrappered_dataset.train_labels)   # type: ignore
            else:
                datas, targets = wrappered_dataset.test_data, np.array(wrappered_dataset.test_labels)     # type: ignore
        else:
            datas = wrappered_dataset.data
            targets = np.array(wrappered_dataset.targets)
        if self._dataidxs is not None:
            datas = datas[self._dataidxs]
            targets = targets[self._dataidxs]
        return datas, targets

    def __getitem__(self, index):
        data, target = self._datas[index], self._targets[index]
        data = Image.fromarray(data)
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target

    def __len__(self):
        if self._dataidxs is None:
            return self._targets.shape[0]
        else:
            return len(self._dataidxs)

    @property
    def data(self):
        return self._datas
    
    @property
    def targets(self):
        return self._targets


if __name__ == "__main__":
    num_classes = 10
    num_dim = 128
    synth = Synthetic(ns=10, num_dim=num_dim, num_classes=num_classes, prob_cluster=[1.0])
    dataset = synth.generator()
    print(dataset)
    # print(synth._prob_cluster)
