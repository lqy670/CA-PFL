#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torchvision.datasets import CIFAR100
from typing import Optional, List, Callable
from dataset.base import WrapperedDataset, WrapperedAllDataset

__all__ = ["CIFAR100Wrapper", "CIFAR100ALLWrapper"]


class CIFAR100Wrapper(WrapperedDataset):
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
        super().__init__(root, train, dataidxs, download, transform, target_transform)
        self._datas, self._targets = self._build_datasets()


class CIFAR100ALLWrapper(WrapperedAllDataset):
    DATASET = CIFAR100

    def __init__(
            self,
            root: str,
            dataidxs: Optional[List[int]] = None,
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ) -> None:
        super().__init__(root, dataidxs, download, transform, target_transform)
        self._datas, self._targets = self._build_datasets()


if __name__ == "__main__":
    import numpy as np
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = CIFAR100ALLWrapper(
        root="../../data/cifar100",
        # train=False,
        # dataidxs=[0, 1],
        download=True,
        transform=transform
    )

    # 3000张图片的mean std
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    # 3000张图片的mean、std
    train_train = next(iter(dataloader))[0]
    print(train_train.shape)
    train_mean = np.mean(train_train.numpy(), axis=(0, 2, 3))
    train_std = np.std(train_train.numpy(), axis=(0, 2, 3))

    print("train_mean:", train_mean)
    print("train_std:", train_std)
