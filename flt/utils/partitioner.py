#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import numpy as np
from decimal import Decimal
from typing import List, Union
from torchvision import transforms
from sklearn.model_selection import train_test_split
from flt import dataset as datastore
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class Partitioner(object):
    def __init__(self, dataset, num: int) -> None:
        self._dataset = dataset
        self._num = num

    def partition(self):
        pass

    def count_target(self):
        # 获取当前的所有类别 id 以及每个类别的数量
        return np.unique(self._dataset.targets, return_index=False, return_counts=True)


# origin iid
class IIDPartitioner(Partitioner):
    def __init__(self, dataset, num: int) -> None:
        super().__init__(dataset, num)

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        targets = self._dataset.targets
        # 获取当前的label，以及每个label的样本数量
        label, counts = self.count_target()
        # 类别的数量
        K = len(label)
        slices = {k: [] for k in range(self._num)}
        for class_idx in range(K):
            # 取出所有的当前的类别的样本下标
            sample_idxs = np.where(targets == class_idx)[0]
            np.random.shuffle(sample_idxs)
            class_counts = counts[class_idx]
            # 每份切分的数据样本长度
            step = (class_counts // self._num) if class_counts % self._num == 0 else (class_counts // self._num) + 1
            for i, start in enumerate(range(0, class_counts, step)):
                if i not in slices.keys():
                    raise RuntimeError("Client Id is not consistent with slice number !!!")
                slices[i].extend(sample_idxs[start: (start + step)])
        return slices


# balanced iid
class IIDPartitioner_balance(Partitioner):
    def __init__(self, dataset, num: int) -> None:
        super().__init__(dataset, num)

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        targets = self._dataset.targets
        num_samples = len(targets)
        num_sample_per_client = int(num_samples / self._num)
        client_sample_nums = (np.ones(self._num) * num_sample_per_client).astype(int)
        rand_perm = np.random.permutation(num_samples)  # 对0~n随机排序
        num_cumsum = np.cumsum(client_sample_nums).astype(int)  # [6k,12k,18k,...,6w]
        client_indices_pair = [(cid, idxs) for cid, idxs in enumerate(np.split(rand_perm, num_cumsum)[:-1])]
        return dict(client_indices_pair)


# unbalanced iid
# Sample numbers for clients are drawn from Log-Normal distribution
class IIDPartitioner_unbalance(Partitioner):
    def __init__(self, dataset, num: int) -> None:
        super().__init__(dataset, num)

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        targets = self._dataset.targets
        num_samples = len(targets)
        num_sample_per_client = int(num_samples / self._num)
        client_sample_nums = np.random.lognormal(mean=np.log(num_sample_per_client), sigma=0.3, size=self._num)
        client_sample_nums = (client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0

        # Add/Substract the excedd number starting from first client
        if diff != 0:
            for cid in range(self._num):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break

        rand_perm = np.random.permutation(num_samples)  # 对0~n随机排序
        num_cumsum = np.cumsum(client_sample_nums).astype(int)
        client_indices_pair = [(cid, idxs) for cid, idxs in enumerate(np.split(rand_perm, num_cumsum)[:-1])]
        return dict(client_indices_pair)


# 迪利克雷分布 noniid (unbalanced)
class DirichletPartitioner(Partitioner):
    def __init__(self, dataset, num: int, alpha: float = 0.5, min_require_size: int = 10) -> None:
        super().__init__(dataset, num)
        self._alpha = alpha
        self._min_require_size = min_require_size

    def partition(self):
        """
        数据切分
        :raises RuntimeError: 节点数量与实际生成的切片数量不一致抛出异常
        :return dict[int, list[int]]: 每个客户端切分后的数据样本下标
        """
        min_size = 0
        targets = self._dataset.targets
        # 获取当前的label，以及每个label的样本数量
        label, _ = self.count_target()
        # 类别的数量
        K = len(label)
        N = targets.shape[0]
        slices = {k: [] for k in range(self._num)}
        idx_batch = [[] for _ in range(self._num)]
        while min_size < self._min_require_size:
            idx_batch = [[] for _ in range(self._num)]
            for k in range(K):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                props = np.random.dirichlet(np.repeat(self._alpha, self._num))
                props = np.array([p * (len(idx_j) < N / self._num) for p, idx_j in zip(props, idx_batch)])
                props = props / props.sum()
                props = (np.cumsum(props) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, props))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(self._num):
            np.random.shuffle(idx_batch[j])
            slices[j] = idx_batch[j]
        return slices


# 病态分布 noniid (balanced)
class PathologicalPartitioner(Partitioner):
    def __init__(self, dataset, num: int, num_classes_per_client: int = 2) -> None:
        super().__init__(dataset, num)
        self._num_classes_per_client = num_classes_per_client

    def _sort_samples_with_label(self, targets: np.ndarray, label: List[int]):
        """
        将样本按照标签的顺序进行排序
        :return List[int]: 获得排序后的样本下标的列表
        """
        # K 个类别
        K = len(label)
        idxlst = []
        for k in range(K):
            kth_idxs = np.where(targets == k)[0]
            idxlst.extend(kth_idxs.tolist())
        return idxlst

    def _divided(self, idxlst: Union[List[int], List[List[int]]], gn: int):
        """
        将 idxlst 下标的列表分成 gn 组
        :param List[int] idxlst: 下标的列表
        :param int gn: 组数
        """
        ns = len(idxlst)
        gs = int(ns // gn)
        # 有多少组的样本数量多一个
        gnp1 = ns - gs * gn
        # 有多少组的样本数量不多
        gnp0 = gn - gnp1
        shards = []
        for k in range(gnp0):
            s, e = k * gs, (k + 1) * gs
            shards.append(idxlst[s:e])

        start = gnp0 * gs
        for k in range(gnp1):
            s, e = k * (gs + 1), (k + 1) * (gs + 1)
            shards.append(idxlst[(start + s):(start + e)])
        return shards

    def partition(self):
        ### step 1 将样本按照标签的顺序进行排序
        targets = self._dataset.targets
        label, _ = self.count_target()
        idxlst = self._sort_samples_with_label(targets, label)
        ### step 2 将得到的样本下标列表随机分成若干份 shards
        # 分成 M 份
        M = self._num * self._num_classes_per_client
        shards = self._divided(idxlst, M)

        ### step 3 划分，得到 N 份联邦数据集
        slices = {k: [] for k in range(self._num)}
        np.random.shuffle(shards)
        groups = self._divided(shards, self._num)
        # 合并每个 shard 切片
        for idx, group in enumerate(groups):
            slices[idx] = [*group[0], *group[1]]
        return slices


def show_distribution(dataset, dataidx: dict, imp: str):
    from matplotlib import pyplot as plt

    K = len(dataidx.keys())
    targets = dataset.targets
    # plt.figure(figsize=(8, 5))
    plt.figure()
    plt.hist(
        [targets[v] for _, v in dataidx.items()],
        stacked=True,
        bins=np.arange(min(targets) - 0.5, max(targets) + 1.5, 1),
        label=["Client {}".format(i) for i in range(K)],  # type: ignore
        rwidth=0.4,
    )
    # x_major_locator = plt.MultipleLocator(3) # type: ignore
    # ax = plt.gca()
    # ax为两条坐标轴的实例
    # ax.xaxis.set_major_locator(x_major_locator)
    plt.xlabel("Classes")
    plt.ylabel("Volume")
    # plt.xticks(range(0, 10, 1), range(10), rotation=20)
    plt.xticks(range(0, len(dataset.classes), 1), dataset.classes, rotation=20, fontsize=10)
    plt.legend(loc="lower center", ncol=int(K // 2), bbox_to_anchor=(0.5, -0.3), borderaxespad=0.)
    plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.25)
    plt.show()
    plt.savefig(imp, dpi=600)
    pass


def restruce_data_from_dataidx(datadir: str, dataset: str, dataidx_map: dict, n_clusters):
    """
    重新依据dataidx生成每个节点的数据切分
    :param str datadir: 数据集路径
    :param str dataset: 数据集名称
    :param dict dataidx_map: 切分的下标字典
    :param n_clusters: 社区聚类数
    :return dict: train_datasets, test_dataset, global_test_dataset 拆分后的训练集,测试集与全局测试集
    """
    train_datasets = {}
    test_datasets = {}
    train_slices = {}
    test_slices = {}

    if dataset == "cifar10":
        # transform = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #         std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        #     )
        # ])
        # 划分训练测试集
        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        # 类别数
        n_client_per_cluster = 10 // n_clusters
        n_degree_per_cluster = 360 // n_clusters
        n_degree_last = n_degree_per_cluster * (n_clusters - 1)
        cnt_cluster = 0
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        for idx, dataidx in train_slices.items():
            if cnt_cluster <= n_clusters - 2:
                degree = n_degree_per_cluster * cnt_cluster
            else:
                degree = n_degree_last
            from flt.utils.gaussian_blur import GaussianBlur
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                # transforms.RandomRotation((90, 90)),
                # transforms.RandomApply([color_jitter], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                GaussianBlur(kernel_size=int(0.1 * 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])
            train_datasets[idx] = datastore.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                              transform=transform)
            if (idx + 1) % n_client_per_cluster == 0:
                cnt_cluster += 1

        cnt_cluster = 0
        for idx, dataidx in test_slices.items():
            if cnt_cluster <= n_clusters - 2:
                degree = n_degree_per_cluster * cnt_cluster
            else:
                degree = n_degree_last
            transform = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((32, 32)),
                # transforms.RandomRotation((degree, degree)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),

                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])
            test_datasets[idx] = datastore.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                             transform=transform)
            if (idx + 1) % n_client_per_cluster == 0:
                cnt_cluster += 1

    elif dataset == "cifar100":
        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        print("[Data Cluster] Client 0-3:rotate 不变 -- Client 4-6:rotate 120 -- Client 7-9:rotate 240")
        for idx, dataidx in train_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 120
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                    )
                ])
            else:  # 240
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                    )
                ])
            train_datasets[idx] = datastore.CIFAR100ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                            transform=transform)
        for idx, dataidx in test_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 120
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                    )
                ])
            else:  # 240
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
                    )
                ])
            test_datasets[idx] = datastore.CIFAR100ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                           transform=transform)

    elif dataset == "mnist":
        # transform = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.1309001],
        #         std=[0.28928837]
        #     )
        # ])
        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        print("[Data Cluster] Client 0-3:rotate 不变 -- Client 4-6:rotate 120 -- Client 7-9:rotate 240")
        for idx, dataidx in train_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.1309001],
                        std=[0.28928837]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 120
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.1309001],
                        std=[0.28928837]
                    )
                ])
            else:  # 240
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.1309001],
                        std=[0.28928837]
                    )
                ])
            train_datasets[idx] = datastore.MNISTALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                            transform=transform)
        for idx, dataidx in test_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.1309001],
                        std=[0.28928837]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 120
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.1309001],
                        std=[0.28928837]
                    )
                ])
            else:  # 240
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.1309001],
                        std=[0.28928837]
                    )
                ])
            test_datasets[idx] = datastore.MNISTALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                           transform=transform)

    elif dataset == "fashionmnist":
        # transform = transforms.Compose([
        #     transforms.Resize((32, 32)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.2855559],
        #         std=[0.3384841]
        #     )
        # ])
        # for idx, dataidx in dataidx_map.items():
        #     train_datasets[idx] = datastore.FMNISTWrapper(root=datadir, train=True, dataidxs=dataidx, download=False,
        #                                                   transform=transform)
        # test_dataset = datastore.FMNISTWrapper(root=datadir, train=False, dataidxs=None, download=False,
        #                                        transform=transform)

        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        print("[Data Cluster] Client 0-3:rotate 不变 -- Client 4-6:rotate 120 -- Client 7-9:rotate 240")
        for idx, dataidx in train_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 120
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            else:  # 240
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            train_datasets[idx] = datastore.FMNISTALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                          transform=transform)
        for idx, dataidx in test_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 120
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            else:  # 240
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            test_datasets[idx] = datastore.FMNISTALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                         transform=transform)

    #     train_datasets[idx] = datastore.MNISTWrapper(root=datadir, train=True, dataidxs=dataidx, download=False,
    #                                                  transform=transform)
    # test_dataset = datastore.MNISTWrapper(root=datadir, train=False, dataidxs=None, download=False,
    #                                       transform=transform)

    return train_datasets, test_datasets


class CustomSubset(Dataset):
    def __init__(self, dataset, subset_transform=None):
        self.dataset = dataset
        self.subset_transform = subset_transform

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y

    def __len__(self):
        return len(self.dataset)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = VisionF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


if __name__ == "__main__":
    import sys
    import os
    import pickle
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as VisionF

    sys.path.append(".")
    from flt.dataset import cifar10, MNISTALLWrapper, cifar100

    np.random.seed(26)

    # TODO CIFAR10
    dataset = cifar10.CIFAR10ALLWrapper(
        root="../../data/cifar10",
        # train=True,
        download=False,
    )
    # TODO MNIST
    # dataset = MNISTALLWrapper(
    #     root="../../data/mnist",
    #     # dataidxs=[0, 1],
    #     download=False,
    #     # transform=transform
    # )

    # 缓存当前切分的样本切分数据
    dataset_cache_dir = os.path.join("cache", "cifar10")
    if not os.path.exists(dataset_cache_dir):
        os.makedirs(dataset_cache_dir)
    cache_filename = f"practical_non_iid_d_0.3_10.dataidx"
    dataset_cache = os.path.join(dataset_cache_dir, cache_filename)
    if not os.path.exists(dataset_cache):
        print(f"{cache_filename} file not exists, generate it")
        # 数据集切分
        # partitioner = IIDPartitioner(dataset, num=10)
        # partitioner = IIDPartitioner_balance(dataset, num=10)
        partitioner = DirichletPartitioner(dataset, alpha=0.3, num=10)
        # partitioner = PathologicalPartitioner(dataset, num=10)
        # partitioner = IIDPartitioner_unbalance(dataset, num=10)
        dataidx_map = partitioner.partition()
        # 保存当前的切分样本下标
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataidx_map, f)
    else:
        print(f"{cache_filename} file exists, load it")
        with open(dataset_cache, "rb") as f:
            dataidx_map = pickle.load(f)
    dataidx_map_count = {k: len(v) for k, v in dataidx_map.items()}
    print(f"all clients samples dataidx {dataidx_map_count}")
    train_datasets, test_dataset = restruce_data_from_dataidx(datadir="../../data/cifar10", dataset="cifar10",
                                                              dataidx_map=dataidx_map, n_clusters=3)
    print("train/test datas and labels for all clients:")
    for k in range(10):
        print(
            f"clients {k}: train datas: {train_datasets[k].cls_num_map}, test datas: {test_dataset[k].cls_num_map}")
    print("---------------------------------------------")
    dataloader = DataLoader(dataset=train_datasets[8], batch_size=32, shuffle=True)

    for batch in dataloader:
        img1, img2 = batch
        # (img1, img2, _), label = batch
        break

    img_grid = make_grid(img1, normalize=True)
    show(img_grid)

    # for k in range(10):
    #     print(
    #         f"clients {k}: train datas: {len(train_datasets[k].cls_num_map)}, test datas: {len(test_dataset[k].cls_num_map)}")
    # print("---------------------------------------------")
    # for k in range(10):
    #     train = set(train_datasets[k].cls_num_map.keys())
    #     test = set(test_dataset[k].cls_num_map.keys())
    #     if test.issubset(train):
    #         print(1)
    #     else:
    #         print(0)

    # partitioner = DirichletPartitioner(dataset, 10)
    # partitioner = DirichletPartitioner(dataset, 10, alpha=0.5)
    # partitioner = PathologicalPartitioner(dataset, 10, num_classes_per_client=20)
    # dataidx_map = partitioner.partition()
    # show_distribution(dataset, dataidx_map, "./test.png")
