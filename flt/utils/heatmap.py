"""
@author: 670
@school: upc
@file: data_heatmap.py
@date: 2022/11/15 17:32
@desc: 数据集分布热力图
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
# from flt.dataset import Cifar10Wrapper, Cifar10_All_Wrapper
from sklearn.model_selection import train_test_split
from flt.dataset import cifar10

def restruce_data_from_dataidx(datadir: str, dataidx_map: dict):
    """
    重新依据dataidx生成每个节点的数据切分
    :param str datadir: 数据集路径
    :param str dataset: 数据集名称
    :param dict dataidx_map: 切分的下标字典
    :return dict: 拆分后的每个切分数据集
    """
    train_datasets = {}
    test_datasets = {}
    train_slices = {}
    test_slices = {}
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
    ])
    # 划分训练测试集
    for idx, dataidx in dataidx_map.items():
        train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                               shuffle=True)
    for idx, dataidx in train_slices.items():
        train_datasets[idx] = cifar10.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                  transform=transform)
    for idx, dataidx in test_slices.items():
        test_datasets[idx] = cifar10.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                 transform=transform)
    global_test_dataset = cifar10.CIFAR10Wrapper(root=datadir, train=False, dataidxs=None, download=False,
                                         transform=transform)

    return train_datasets, test_datasets, global_test_dataset


if __name__ == '__main__':
    path = '../../cache/cifar10/balanced_old_iid_10.dataidx'
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    # 分布不平衡热力图
    plt.subplot(2, 2, 1)
    with open(path, "rb") as f:
        dataidx_map = pickle.load(f)
    train_datasets, test_dataset, global_test_dataset = restruce_data_from_dataidx(datadir='../../data/cifar10',
                                                                                   dataidx_map=dataidx_map)
    cls_num_maps = {k: d.cls_num_map for (k, d) in train_datasets.items()}
    # print(cls_num_maps)
    nparray = np.zeros((10, 10))
    for k, v in cls_num_maps.items():
        for a, b in v.items():
            nparray[a][k] = b
    nparray = nparray.astype(np.int64)
    sns.set_theme()
    ax = sns.heatmap(nparray, fmt="d", cmap="YlGnBu", annot=False, annot_kws={"fontsize": 10}).invert_yaxis()
    plt.xlabel("Client ID",fontsize=15)
    plt.ylabel("Class ID",fontsize=15)
    plt.title("IID", pad=15,fontsize=20)

    # 类别不平衡热力图
    path2 = '../../cache/cifar10/practical_non_iid_d_0.7_10.dataidx'
    plt.subplot(2, 2, 2)
    with open(path2, "rb") as f:
        dataidx_map = pickle.load(f)
    train_datasets, test_dataset, global_test_dataset = restruce_data_from_dataidx(datadir='../../data/cifar10',
                                                                                   dataidx_map=dataidx_map)
    cls_num_maps = {k: d.cls_num_map for (k, d) in train_datasets.items()}
    # print(cls_num_maps)
    nparray = np.zeros((10, 10))
    for k, v in cls_num_maps.items():
        for a, b in v.items():
            nparray[a][k] = b
    nparray = nparray.astype(np.int64)
    sns.set_theme()
    ax = sns.heatmap(nparray, fmt="d", cmap="YlGnBu", annot=False, annot_kws={"fontsize": 10}).invert_yaxis()
    plt.xlabel("Client ID",fontsize=15)
    plt.ylabel("Class ID",fontsize=15)
    plt.title("Non-IID α=0.7", pad=15,fontsize=20)

    # 类别不平衡热力图
    path2 = '../../cache/cifar10/practical_non_iid_d_0.5_10.dataidx'
    plt.subplot(2, 2, 3)
    with open(path2, "rb") as f:
        dataidx_map = pickle.load(f)
    train_datasets, test_dataset, global_test_dataset = restruce_data_from_dataidx(datadir='../../data/cifar10',
                                                                                   dataidx_map=dataidx_map)
    cls_num_maps = {k: d.cls_num_map for (k, d) in train_datasets.items()}
    # print(cls_num_maps)
    nparray = np.zeros((10, 10))
    for k, v in cls_num_maps.items():
        for a, b in v.items():
            nparray[a][k] = b
    nparray = nparray.astype(np.int64)
    sns.set_theme()
    ax = sns.heatmap(nparray, fmt="d", cmap="YlGnBu", annot=False, annot_kws={"fontsize": 10}).invert_yaxis()
    plt.xlabel("Client ID", fontsize=15)
    plt.ylabel("Class ID", fontsize=15)
    plt.title("Non-IID α=0.5", pad=15, fontsize=20)

    # 类别不平衡热力图
    path2 = '../../cache/cifar10/practical_non_iid_d_0.3_10.dataidx'
    plt.subplot(2, 2, 4)
    with open(path2, "rb") as f:
        dataidx_map = pickle.load(f)
    train_datasets, test_dataset, global_test_dataset = restruce_data_from_dataidx(datadir='../../data/cifar10',
                                                                                   dataidx_map=dataidx_map)
    cls_num_maps = {k: d.cls_num_map for (k, d) in train_datasets.items()}
    # print(cls_num_maps)
    nparray = np.zeros((10, 10))
    for k, v in cls_num_maps.items():
        for a, b in v.items():
            nparray[a][k] = b
    nparray = nparray.astype(np.int64)
    sns.set_theme()
    ax = sns.heatmap(nparray, fmt="d", cmap="YlGnBu", annot=False, annot_kws={"fontsize": 10}).invert_yaxis()
    plt.xlabel("Client ID", fontsize=15)
    plt.ylabel("Class ID", fontsize=15)
    plt.title("Non-IID α=0.3", pad=15, fontsize=20)

    plt.show()
