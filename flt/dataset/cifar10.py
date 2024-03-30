#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torchvision.datasets import CIFAR10
from typing import Optional, List, Callable
from dataset.base import WrapperedDataset,WrapperedAllDataset

__all__ = ["CIFAR10Wrapper", "CIFAR10ALLWrapper"]


class CIFAR10Wrapper(WrapperedDataset):
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
        super().__init__(root, train, dataidxs, download, transform, target_transform)
        self._datas, self._targets = self._build_datasets()


class CIFAR10ALLWrapper(WrapperedAllDataset):
    DATASET = CIFAR10

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
    import numpy as np
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as VisionF

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize(
        #     mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #     std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        # )
    ])
    dataset = CIFAR10ALLWrapper(
        root="../../data/cifar10",
        # train=True,
        # dataidxs=[0, 1],
        download=False,
        transform=transform
    )

    # 3000张图片的mean std
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    for batch in dataloader:
        img1, img2 = batch
        # (img1, img2, _), label = batch
        break

    img_grid = make_grid(img1, normalize=True)
    show(img_grid)

    # 3000张图片的mean、std
    # train_train = next(iter(dataloader))[0]
    # print(train_train.shape)
    # train_mean = np.mean(train_train.numpy(), axis=(0, 2, 3))
    # train_std = np.std(train_train.numpy(), axis=(0, 2, 3))
    #
    # print("train_mean:", train_mean)
    # print("train_std:", train_std)
