import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from typing import Optional, List, Callable
from torchvision.datasets import CIFAR10

if __name__ == '__main__':
    wrappered_train_dataset = CIFAR10(root="../../data/cifar10", train=True, transform=None,
                                           target_transform=None, download=False)
    wrappered_test_dataset = CIFAR10(root="../../data/cifar10", train=False, transform=None,
                                          target_transform=None, download=False)
    train_datas, train_targets = wrappered_train_dataset.data, np.array(
        wrappered_train_dataset.targets)  # type: ignore
    test_datas, test_targets = wrappered_test_dataset.data, np.array(
        wrappered_test_dataset.targets)  # type: ignore
    # 拼接
    datas = np.append(train_datas, test_datas, axis=0)
    targets = np.append(train_targets, test_targets, axis=0)
    # 获取当前所有类别编号及其数量
    # (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))
    print(np.unique(targets, return_index=False, return_counts=True))

