#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from typing import Optional, List, Callable
from torchvision.datasets import CIFAR10,CIFAR100


# 训练集/测试集
class WrapperedDataset(data.Dataset):
    DATASET = CIFAR100

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
                datas, targets = wrappered_dataset.train_data, np.array(wrappered_dataset.train_labels)  # type: ignore
            else:
                datas, targets = wrappered_dataset.test_data, np.array(wrappered_dataset.test_labels)  # type: ignore
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
    def datas(self):
        return self._datas

    @property
    def targets(self):
        return self._targets


# 训练集+测试集
class WrapperedAllDataset(data.Dataset):
    DATASET = CIFAR100

    def __init__(
            self,
            root: str,
            dataidxs: Optional[List[int]] = None,
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        self._root = root
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
        wrappered_train_dataset = self.DATASET(root=self._root, train=True, transform=self._transform,
                                               target_transform=self._target_transform, download=self._download)
        wrappered_test_dataset = self.DATASET(root=self._root, train=False, transform=self._transform,
                                              target_transform=self._target_transform, download=self._download)
        self._classes = wrappered_train_dataset.classes
        if torchvision.__version__ == '0.2.1':
            train_datas, train_targets = wrappered_train_dataset.train_data, np.array(
                wrappered_train_dataset.train_labels)  # type: ignore
            test_datas, test_targets = wrappered_test_dataset.test_data, np.array(
                wrappered_test_dataset.test_labels)  # type: ignore
        else:
            train_datas, train_targets = wrappered_train_dataset.data, np.array(
                wrappered_train_dataset.targets)  # type: ignore
            test_datas, test_targets = wrappered_test_dataset.data, np.array(
                wrappered_test_dataset.targets)  # type: ignore
        datas = np.append(train_datas,test_datas,axis=0)
        targets = np.append(train_targets,test_targets,axis=0)
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
    dataset = WrapperedAllDataset(
        root="../../data/cifar100",
        # train=True,
        # dataidxs=[0, 1],
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    print(len(dataset.data))
    print(len(dataset.targets))
    # loader = data.DataLoader(
    #     dataset=dataset,
    #     shuffle=True,
    #     batch_size=4
    # )
    # for (datas, targets) in loader:
    #     print(datas, targets)
