#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Callable
from torchvision.datasets import MNIST, utils
from dataset.base import WrapperedDataset

__all__ = ["FEMNISTWrapper"]


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """
    resources = [
        # ('https://ghproxy.com/https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz',
        # '59c65cec646fc57fe92d27d83afdf0ed')
        ("https://s3.amazonaws.com/nist-srd/SD19/by_class.zip", ""),
        ("https://s3.amazonaws.com/nist-srd/SD19/by_class.zip", ""),
    ]

    splits = ("by_class", "by_write")

    def __init__(
            self,
            root: str,
            train: bool=True,
            transform: Optional[Callable]=None,
            target_transform: Optional[Callable]=None,
            download: bool=False
        ) -> None:
        super(MNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(os.path.join(self.processed_folder, data_file))
        print(len(self.users_index))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='F')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def download(self):
        """Download the FEMNIST data if it doesn't exist in processed_folder already."""
        import shutil

        if self._check_exists():
            return

        self._makedir_exist_ok(self.raw_folder)
        self._makedir_exist_ok(self.processed_folder)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            utils.download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')
        shutil.move(os.path.join(self.raw_folder, self.training_file), self.processed_folder)
        shutil.move(os.path.join(self.raw_folder, self.test_file), self.processed_folder)

    def _makedir_exist_ok(self, root):
        if not os.path.exists(root):
            os.makedirs(root)

    def _check_exists(self) -> bool:
        return all(
            utils.check_integrity(os.path.join(self.raw_folder, os.path.basename(url)))
            for url, _ in self.resources
        )

class FEMNISTWrapper(WrapperedDataset):
    DATASET = FEMNIST
    def __init__(
            self, 
            root: str, 
            train: bool = True, 
            dataidxs: Optional[List[int]] = None, 
            download: bool = False, 
            transform = None, 
            target_transform = None
        ) -> None:
        super().__init__(root, train, dataidxs, download, transform, target_transform)
        self._datas, self._targets = self._build_datasets()

    def __getitem__(self, index):
        data, target = self._datas[index], self._targets[index]
        target = target.astype(np.int64)
        data = (data * 255).numpy().astype('uint8')
        data = Image.fromarray(data, mode="L")
        if self._transform is not None:
            data = self._transform(data)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return data, target


if __name__ == "__main__":
    femnist = FEMNIST(
        root="../../data/femnist",
        train=True,
        download=True
    )
    print(femnist.data.shape, femnist.targets.shape, femnist.classes)

    import numpy as np
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = FEMNISTWrapper(
        root="../../data/femnist",
        train=True,
        # dataidxs=[0, 1],
        download=True,
        transform=transform
    )
    
    # 3000张图片的mean std
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=True)
    # 3000张图片的mean、std
    train_train, label = next(iter(dataloader))
    print(train_train.shape)
    print(label[0])
    train_mean = np.mean(train_train.numpy(), axis=(0, 2, 3))
    train_std = np.std(train_train.numpy(), axis=(0, 2, 3))

    print("train_mean:", train_mean)
    print("train_std:", train_std)
    pass
