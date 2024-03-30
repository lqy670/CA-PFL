#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from PIL import Image
from torchvision.datasets import FashionMNIST
from typing import Optional, List, Callable
from dataset.base import WrapperedDataset, WrapperedAllDataset

__all__ = ["FMNISTWrapper", "FASHIONMNISTALLWrapper"]


class FMNISTWrapper(WrapperedDataset):
    DATASET = FashionMNIST

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

    def __getitem__(self, index):
        data, target = self._datas[index], self._targets[index]
        data = Image.fromarray(data.numpy(), mode="L")
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target


class FASHIONMNISTALLWrapper(WrapperedAllDataset):
    DATASET = FashionMNIST

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

    def __getitem__(self, index):
        data, target = self._datas[index], self._targets[index]
        data = Image.fromarray(data, mode="L")
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target


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
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as VisionF
    import numpy as np
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid

    # dataset = FMNISTWrapper(
    #     root="../../data/fashionmnist",
    #     train=False,
    #     # dataidxs=[0, 1],
    #     download=True,
    # )
    # print(dataset.datas.shape, dataset.targets.shape, dataset.classes)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[0.13066064],
        #     std=[0.30810764]
        # )
    ])
    dataset = FMNISTALLWrapper(
        root="../../data/fashionmnist",
        # dataidxs=[0, 1],
        download=False,
        transform=transform
    )

    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        img1, img2 = batch
        # (img1, img2, _), label = batch
        break

    img_grid = make_grid(img1, normalize=True)
    show(img_grid)



    # 3000张图片的mean std
    # dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    # # 3000张图片的mean、std
    # train_train, label = next(iter(dataloader))
    # print(train_train.shape)
    # print(label[0].dtype)
    # train_mean = np.mean(train_train.numpy(), axis=(0, 2, 3))
    # train_std = np.std(train_train.numpy(), axis=(0, 2, 3))
    #
    # print("train_mean:", train_mean)
    # print("train_std:", train_std)
