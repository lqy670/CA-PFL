#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torchvision
import numpy as np
from PIL import Image
from torchvision.datasets import EMNIST
from typing import Optional, List, Callable
from dataset.base import WrapperedDataset

__all__ = ["EMNISTWrapper"]


class EMNISTWrapper(WrapperedDataset):
    DATASET = EMNIST
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
    
    def _build_datasets(self):
        wrappered_dataset = self.DATASET(
            root=self._root, 
            split="byclass",
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
        data = Image.fromarray(data.numpy(), mode="L")
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target

    
if __name__ == "__main__":
    import numpy as np
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = EMNISTWrapper(
        root="data/emnist",
        train=False,
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
