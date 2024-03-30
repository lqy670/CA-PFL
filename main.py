#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import ast
import torch
import random
import pickle
import logging
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

from flt import network
from torchvision import transforms
from argparse import ArgumentParser

from flt.utils.gaussian_blur import GaussianBlur
from flt.utils.partitioner import IIDPartitioner, DirichletPartitioner, PathologicalPartitioner, \
    IIDPartitioner_balance, IIDPartitioner_unbalance
from flt.algorithms import FedAvg, FedProx, MOON, FedSL, FedBN, QSGD, FedPer, FedASL, CPFL, MOON2, MOON_PER, MOON_PER2, \
    CPFL2, FedCommunity, FedCC,FedTest,ClusterFed,FedRep,FedBabu,PerCFL,Local
# from flt.dataset import CIFAR10Wrapper, CIFAR100Wrapper, MNISTWrapper, \
#     FEMNISTWrapper, EMNISTWrapper, FashionMNISTWrapper, ShakeSpeare
from flt import dataset as datastore


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-b", "--backbone", default="SimpleCNN", type=str, choices=[
        "SimpleCNN", "resnet9", "resnet34", "resnet50", "shufflenet", "chargru"
    ], help="the network/model for experiment")
    parser.add_argument("--net_config", default={}, type=lambda x: list(map(int, x.split(', '))),
                        help="the federated learning network config")

    parser.add_argument("--alg_config", default={}, type=ast.literal_eval,
                        help="the federated learning algorithm config")

    parser.add_argument("-d", "--dataset", default="cifar10", type=str, choices=[
        "cifar10", "cifar100", "mnist", "fashionmnist", "shakespeare"
    ], help="the dataset for training Federated Learning")
    parser.add_argument("--alpha", default=0.5, type=float,
                        help="the dirichlet ratio for dataset split to train Federated Learning")
    parser.add_argument("--datadir", default="./data/", type=str, help="the dataset dir")
    parser.add_argument('--partition', default="practical-non-iid", type=str,
                        choices=["balanced-iid", "unbalanced-iid", "practical-non-iid", "pathological-non-iid"],
                        help="the data partitioning strategy")

    parser.add_argument("-lr", "--learning_rate", default=0.01, type=float, help="the optimizer learning rate")
    parser.add_argument("-bs", "--batch_size", default=16, type=int,
                        help="the batch size for client local epoch training in federated learning")
    parser.add_argument("-wd", "--weight_decay", default=1e-5, type=float,
                        help="the weight decay for optimizer in federated learning")
    parser.add_argument("-optim", "--optim_name", default="sgd", type=str, choices=["sgd", "adam", "amsgrad"],
                        help="the optimizer for client local epoch training in federated learning")
    parser.add_argument("-mu", "--mu", default=1, type=float, help="the mu for fedprox in federated learning")
    parser.add_argument("-temperature", "--temperature", default=0.5, type=float, help="the temperature for MOON")
    parser.add_argument("-n", "--n_parties", default=10, type=int, help="total client numbers in federated learning")
    parser.add_argument("-nk", "--nk_parties", default=10, type=int,
                        help="client numbers for aggregation per communication round in federated learning")
    parser.add_argument("-k", "--n_clusters", default=3, type=int, help="number of clustered categories")
    parser.add_argument("-p", "--p", default=0.9, type=float, help="the ema for fedcc")
    parser.add_argument("-split", "--split", default=3, type=int, help="the split layer for fedc")
    parser.add_argument("-end", "--end", default=35, type=int, help="the end layer for fedc")


    parser.add_argument("--epochs", default=3, type=int, help="the federated learning client local epoch for training")
    parser.add_argument("--rounds", default=50, type=int, help="the federated learning communication rounds")
    parser.add_argument("--alg", default="fedcom", type=str, choices=[
        "fedavg", "fedprox", "moon", "fedsl", "fedbn", "qsgd", "fedper", "fedasl", "cpfl", "moon2", "moon_per",
        "moon_per2", "cpfl2", "fedcom", "fedcc","fedtest","clusterfed","fedrep","fedbabu","perCFL","local"
    ], help="the federated learning algorithm")

    parser.add_argument("--savedir", default="exps", type=str,
                        help="the federated learning algorithm experiment save dir")
    return parser.parse_args()


def init_logger(savedir: str, filename: str):
    # 初始化日志配置模块
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    log_path = os.path.join(savedir, f"{filename}.log")
    logging.basicConfig(
        filename=log_path,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w'
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def init_nets(backbone: str, n: int, net_config: dict = {}):
    """
    初始化所有的节点模型
    :param str backbone: 节点模型的名称, 需要和网络的名称一致
    :param int n: 总共的节点数量
    :param dict net_config: 节点网络模型初始化时候的配置参数, defaults to {}
    :return dict: 节点模型的字典
    """
    try:
        initalizer = getattr(network, backbone)
        clients = {}
        for idx in range(n):
            clients[idx] = initalizer(**net_config)
        return clients
    except AttributeError as e:
        logging.info(f"Network {backbone} can not found !!!")
        return


def init_datasets(datadir: str, dataset: str, partition: str, n_parties: int, n_clusters, alpha: float = 0.5):
    """
    初始化训练数据集与测试数据集, 并将训练数据集分成若干份
    :param str datadir: 数据路径
    :param str dataset: 数据集名称
    :param str partition: 拆分数据集策略
    :param int n_parties: 数据集划分的个数
    :param float alpha: dirichlet拆分浓度参数, default for 0.5
    :return tuple: 拆分训练数据集, 测试数据集
    """
    datadir = os.path.join(datadir, dataset)
    # 这些数据集需要自定义拆分方法
    if dataset in ["cifar10", "cifar100", "mnist", "fashionmnist", "femnist", "emnist"]:
        dn = f"{dataset.upper()}ALLWrapper"
        # dn_global_test = f"{dataset.upper()}Wrapper"
        # 获取当前的数据集的 wrapped 类
        WrapperedDataSet = getattr(datastore, dn)
        # GlobalTestWrapperedDataSet = getattr(datastore, dn_global_test)
        # 构造原始数据集与全局测试集
        datasets = WrapperedDataSet(root=datadir, download=False)
        # global_test_dataset = GlobalTestWrapperedDataSet(root=datadir, download=False, train=False)

        # 缓存当前切分的样本切分数据
        dataset_cache_dir = os.path.join("cache", dataset)
        if not os.path.exists(dataset_cache_dir):
            os.makedirs(dataset_cache_dir)
        if partition == "balanced-iid":
            cache_filename = f"balanced_iid_{n_parties}.dataidx"
        elif partition == "unbalanced-iid":
            cache_filename = f"unbalanced_iid_{n_parties}.dataidx"
        elif partition == "practical-non-iid":
            cache_filename = f"practical_non_iid_d_{alpha}_{n_parties}.dataidx"
        elif partition == "pathological-non-iid":
            cache_filename = f"pathological_non_iid_d_{n_parties}.dataidx"
        else:
            cache_filename = f"balanced_{n_parties}.dataidx"
        dataset_cache = os.path.join(dataset_cache_dir, cache_filename)
        if not os.path.exists(dataset_cache):
            logging.info(f"{cache_filename} file not exists, generate it")
            # 数据集切分
            if partition == "balanced-iid":
                partitioner = IIDPartitioner_balance(datasets, num=n_parties)
            elif partition == "unbalanced-iid":
                partitioner = IIDPartitioner_unbalance(datasets, num=n_parties)
            elif partition == "practical-non-iid":
                partitioner = DirichletPartitioner(datasets, num=n_parties, alpha=alpha)
            elif partition == "pathological-non-iid":
                partitioner = PathologicalPartitioner(datasets, num=n_parties)
            else:
                partitioner = IIDPartitioner_balance(datasets, num=n_parties)
            dataidx_map = partitioner.partition()
            # 保存当前的切分样本下标
            with open(dataset_cache, "wb") as f:
                pickle.dump(dataidx_map, f)
        else:
            logging.info(f"{cache_filename} file exists, load it")
            with open(dataset_cache, "rb") as f:
                dataidx_map = pickle.load(f)
        dataidx_map_count = {k: len(v) for k, v in dataidx_map.items()}
        logging.info(f"all clients samples dataidx {dataidx_map_count}")
        train_datasets, test_dataset, global_test_dataset = restruce_data_from_dataidx(datadir=datadir, dataset=dataset,
                                                                                       dataidx_map=dataidx_map,
                                                                                       n_clusters=n_clusters)
        logging.info("train/test datas and labels for all clients:")
        for k in range(n_parties):
            logging.info(
                f"clients {k}: train datas: {train_datasets[k].cls_num_map}, test datas: {test_dataset[k].cls_num_map}")
        # cls_num_maps = {k: d.cls_num_map for (k, d) in train_datasets.items()}
        # logging.info(cls_num_maps)

    # 该数据集已经拆分完成，仅需要加载即可
    elif dataset in ["shakespeare"]:
        train_datasets = {}
        if dataset == "shakespeare":
            for k in range(n_parties):
                train_datasets[k] = datastore.ShakeSpeare(
                    train=True, download=True, smpd_un=0.01, smpd_df=1.0, smpd_dt=partition,
                    sept_frac=0.8, sept_by_type="sample", user=k
                )
            test_dataset = datastore.ShakeSpeare(
                train=False, download=True, smpd_un=0.01, smpd_df=1.0, smpd_dt=partition,
                sept_frac=0.8, sept_by_type="sample", user=list(range(n_parties))
            )
        # 其余的自定义数据不支持
        else:
            logging.error(f"Unsupport dataset {dataset}")
            raise RuntimeError(f"Unsupport dataset {dataset}")
    # 其余的为不支持的数据集
    else:
        logging.error(f"Unsupport dataset {dataset}")
        raise RuntimeError(f"Unsupport dataset {dataset}")
    return train_datasets, test_dataset, global_test_dataset


def restruce_data_from_dataidx(datadir: str, dataset: str, dataidx_map: dict, n_clusters):
    """
    重新依据dataidx生成每个节点的数据切分
    :param str datadir: 数据集路径
    :param str dataset: 数据集名称
    :param dict dataidx_map: 切分的下标字典
    :return dict: train_datasets, test_dataset, global_test_dataset 拆分后的训练集,测试集与全局测试集
    """
    train_datasets = {}
    test_datasets = {}
    train_slices = {}
    test_slices = {}

    if dataset == "cifar10":
        # 划分训练测试集
        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        logging.info("[Data Cluster] Client 0-3:rotate 不变 -- Client 4-6:rotate 120 -- Client 7-9:rotate 240")
        # logging.info("[Data Cluster] Client 012:rotate 不变 -- Client 345:rotate 90 -- Client 67:rotate 180 -- Client 89:rotate 270")
        # logging.info("[Data Cluster] Client 0-4:rotate 不变 -- Client 5-9:rotate 180")
        for idx, dataidx in train_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 90
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            else:  # 270
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            train_datasets[idx] = datastore.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                              transform=transform)

        for idx, dataidx in test_slices.items():
            if idx >= 0 and idx < 4:  # 不变
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            elif idx >= 4 and idx < 7:  # 90
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            else:  # 270
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            test_datasets[idx] = datastore.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                             transform=transform)
        # 全局测试集
        global_test_dataset = datastore.CIFAR10Wrapper(root=datadir, train=False, dataidxs=None, download=False,
                                                       transform=transforms.Compose([
                                                           transforms.Resize((32, 32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(
                                                               mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                               std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                                                           )
                                                       ]))
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

        # 全局测试集
        global_test_dataset = datastore.CIFAR100Wrapper(root=datadir, train=False, dataidxs=None, download=False,
                                                        transform=transforms.Compose([
                                                            transforms.Resize((32, 32)),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize(
                                                                mean=[0.5070751592371323, 0.48654887331495095,
                                                                      0.4409178433670343],
                                                                std=[0.2673342858792401, 0.2564384629170883,
                                                                     0.27615047132568404]
                                                            )
                                                        ]))
    elif dataset == "mnist":
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
        # 全局测试集
        global_test_dataset = datastore.MNISTWrapper(root=datadir, train=False, dataidxs=None, download=False,
                                                       transform=transforms.Compose([
                                                           transforms.Resize((32, 32)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(
                                                               mean=[0.1309001],
                                                               std=[0.28928837]
                                                           )
                                                       ]))
    elif dataset == "fashionmnist":
        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        print("[Data Cluster] Client 0-3:rotate 不变 -- Client 4-6:rotate 120 -- Client 7-9:rotate 240")
        # logging.info("[Data Cluster] Client 0-4:rotate 不变 -- Client 5-9:rotate 180")
        # logging.info("[Data Cluster] Client 012:rotate 不变 -- Client 345:rotate 90 -- Client 67:rotate 180 -- Client 89:rotate 270")
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
            elif idx >= 4 and idx < 7:  # 90
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            # elif idx >= 7 and idx < 8:  # 180
            #     transform = transforms.Compose([
            #         transforms.Resize((32, 32)),
            #         transforms.RandomRotation((180, 180)),
            #         transforms.ToTensor(),
            #         transforms.Normalize(
            #             mean=[0.2855559],
            #             std=[0.3384841]
            #         )
            #     ])
            else:  # 270
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            train_datasets[idx] = datastore.FASHIONMNISTALLWrapper(root=datadir, dataidxs=dataidx, download=False,
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
            elif idx >= 4 and idx < 7:  # 90
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((120, 120)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            # elif idx >= 6 and idx < 8:  # 180
            #     transform = transforms.Compose([
            #         transforms.Resize((32, 32)),
            #         transforms.RandomRotation((180, 180)),
            #         transforms.ToTensor(),
            #         transforms.Normalize(
            #             mean=[0.2855559],
            #             std=[0.3384841]
            #         )
            #     ])
            else:  # 270
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((240, 240)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.2855559],
                        std=[0.3384841]
                    )
                ])
            test_datasets[idx] = datastore.FASHIONMNISTALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                         transform=transform)
        # 全局测试集
        global_test_dataset = datastore.FMNISTWrapper(root=datadir, train=False, dataidxs=None, download=False,
                                                     transform=transforms.Compose([
                                                         transforms.Resize((32, 32)),
                                                         transforms.ToTensor(),
                                                         transforms.Normalize(
                                                             mean=[0.2855559],
                                                             std=[0.3384841]
                                                         )
                                                     ]))
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.17359173],
                std=[0.33165216]
            )
        ])
        for idx, dataidx in dataidx_map.items():
            train_datasets[idx] = datastore.EMNISTWrapper(root=datadir, train=True, dataidxs=dataidx, download=False,
                                                          transform=transform)
        test_dataset = datastore.EMNISTWrapper(root=datadir, train=False, dataidxs=None, download=False,
                                               transform=transform)
    elif dataset == "femnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.03649244],
                std=[0.14329645]
            )
        ])
        for idx, dataidx in dataidx_map.items():
            train_datasets[idx] = datastore.FEMNISTWrapper(root=datadir, train=True, dataidxs=dataidx, download=False,
                                                           transform=transform)
        test_dataset = datastore.FEMNISTWrapper(root=datadir, train=False, dataidxs=None, download=False,
                                                transform=transform)
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        for idx, dataidx in dataidx_map.items():
            train_datasets[idx] = datastore.CIFAR10Wrapper(root=datadir, train=True, dataidxs=dataidx, download=False,
                                                           transform=transform)
        test_dataset = datastore.CIFAR10Wrapper(root=datadir, train=False, dataidxs=None, download=False,
                                                transform=transform)
    return train_datasets, test_datasets, global_test_dataset


def init_algorithms(
        algorithm: str, global_net, nets: dict, train_datasets: dict, test_dataset, global_test_dataset,
        nk_parties, E, comm_round, lr, batch_size, weight_decay, optim_name, device, savedir, *args, **kwargs):
    logging.info(f"Load {algorithm.upper()} for training")
    if algorithm == "fedavg":
        trainer = FedAvg(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            device=device, savedir=savedir
        )
    elif algorithm == "local":
        trainer = Local(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            device=device, savedir=savedir
        )
    elif algorithm == "clusterfed":
        trainer = ClusterFed(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            device=device, savedir=savedir
        )
    elif algorithm == "fedtest":
        trainer = FedTest(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            device=device, savedir=savedir
        )
    elif algorithm == "fedbn":
        trainer = FedBN(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            device=device, savedir=savedir
        )
    elif algorithm == "fedprox":
        trainer = FedProx(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, mu=kwargs.get("mu", 0.01),
            device=device, savedir=savedir
        )
    elif algorithm == "moon":
        trainer = MOON(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1),
            temperature=kwargs.get("temperature", 0.5),
            pool_size=kwargs.get("pool_size", 1),
            device=device, savedir=savedir
        )
    elif algorithm == "moon2":
        trainer = MOON2(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1),
            temperature=kwargs.get("temperature", 0.5),
            pool_size=kwargs.get("pool_size", 1),
            device=device, savedir=savedir
        )
    elif algorithm == "moon_per":
        trainer = MOON_PER(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1),
            temperature=kwargs.get("temperature", 0.5),
            pool_size=kwargs.get("pool_size", 1),
            device=device, savedir=savedir
        )
    elif algorithm == "moon_per2":
        trainer = MOON_PER2(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1),
            temperature=kwargs.get("temperature", 0.5),
            pool_size=kwargs.get("pool_size", 1),
            device=device, savedir=savedir
        )
    elif algorithm == "fedcc":
        trainer = FedCC(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 0.2),
            temperature=kwargs.get("temperature", 0.5),
            pool_size=kwargs.get("pool_size", 1),
            p=kwargs.get("p", 0.99),
            device=device, savedir=savedir
        )
    elif algorithm == "cpfl":
        trainer = CPFL(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1), m=kwargs.get("m", 0.99),
            device=device, savedir=savedir
        )
    elif algorithm == "cpfl2":
        trainer = CPFL2(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1), m=kwargs.get("m", 0.99),
            device=device, savedir=savedir
        )
    elif algorithm == "fedcom":
        trainer = FedCommunity(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name,
            mu=kwargs.get("mu", 1),
            temperature=kwargs.get("temperature", 0.5),
            pool_size=kwargs.get("pool_size", 1),
            p=kwargs.get("p", 0.99),
            split=kwargs.get("split", 3),
            end=kwargs.get("end", 35),
            device=device, savedir=savedir
        )
    elif algorithm == "fedrep":
        trainer = FedRep(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, device=device,
            savedir=savedir
        )
    elif algorithm == "fedbabu":
        trainer = FedBabu(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, device=device,
            savedir=savedir
        )
    elif algorithm == "perCFL":
        trainer = PerCFL(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, device=device,
            savedir=savedir
        )
    elif algorithm == "fedsl":
        trainer = FedSL(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, mu=kwargs.get("mu", 0.01),
            q0=kwargs.get("q0", 0.9), s=kwargs.get("s", 5),
            # q0=kwargs.get("q0", 0.95), s=kwargs.get("s", 10),
            p=kwargs.get("p", 3), wc=kwargs.get("wc", False),
            device=device, savedir=savedir
        )
        pass
    elif algorithm in ["fedadaptivesl", "fedasl", "fedckasl"]:
        mu = kwargs.get("mu", 0.01)
        kwargs.pop("mu")
        q0 = kwargs.get("q0", 0.90)
        s = kwargs.get("s", 5)
        if "q0" in kwargs:
            kwargs.pop("q0")
        if "s" in kwargs:
            kwargs.pop("s")
        trainer = FedASL(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, mu=mu,
            # q0=kwargs.get("q0", 0.9), s=kwargs.get("s", 5),
            p=kwargs.get("p", 3), wc=kwargs.get("wc", False), q0=q0, s=s,
            device=device, savedir=savedir, **kwargs
        )
    elif algorithm == "qsgd":
        trainer = QSGD(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, device=device,
            savedir=savedir
        )
        pass
    elif algorithm == "fedper":
        trainer = FedPer(
            global_net=global_net, nets=nets, datasets=train_datasets, test_dataset=test_dataset,
            global_test_dataset=global_test_dataset,
            nk_parties=nk_parties, E=E, comm_round=comm_round,
            lr=lr, batch_size=batch_size, weight_decay=weight_decay, optim_name=optim_name, device=device,
            savedir=savedir
        )
        pass
    else:
        trainer = None
    return trainer


def train(network: str, datadir: str, dataset: str, algorithm: str, partition: str, n_parties: int,
          alpha: float, savedir: str, n_clusters, args):
    # classes
    if dataset in ["cifar10", "mnist", "fashionmnist"]:
        n_classes = 10
    elif dataset == "cifar100":
        n_classes = 100
    elif dataset in ["emnist", "femnist"]:
        n_classes = 62
    elif dataset == "shakespeare":
        n_classes = 80
    else:
        n_classes = 10

    # channel
    if dataset in ["cifar10", "cifar100"]:
        in_channel = 3
    elif dataset in ["mnist", "fashionmnist", "emnist", "femnist"]:
        in_channel = 1
    else:
        in_channel = 3
    # 如果是MOON算法，则由于输出多个值，需要加载不同的模型
    if algorithm == "moon" or algorithm == "moon2" or algorithm == "moon_per" or algorithm == "moon_per2" \
            or algorithm == "cpfl2" or algorithm == "fedcc" or algorithm == "fedcom":
        net_config = {"model_name": f"{network}", "num_classes": n_classes, "in_channel": in_channel}
        network = "ModelFedCon"
    # cpfl算法
    elif algorithm == "cpfl":
        net_config = {"model_name": f"{network}", "num_classes": n_classes, "in_channel": in_channel}
        network = "Modelcpfl"
    else:
        net_config = {"num_classes": n_classes, "in_channel": in_channel}
    # 获取模型
    logging.info(f"Load network: {network}")
    # cpfl需要维护两个全局模型
    if algorithm == "cpfl" or algorithm == "cpfl2":
        global_nets = init_nets(network, 2, net_config)  # global_nets = {0:model0,1:model1}
        if global_nets is None or global_nets.get(0) is None or global_nets.get(1) is None:
            logging.info("Error, initialize global model failed")
            return
    else:
        global_nets = init_nets(network, 1, net_config)
        if global_nets is None or global_nets.get(0) is None:
            logging.info("Error, initialize global model failed")
            return
        global_nets = global_nets[0]
    nets = init_nets(network, n_parties, net_config)
    # 获取训练集、测试集与整体测试集
    train_datasets, test_dataset, global_test_dataset = init_datasets(datadir, dataset, partition=partition,
                                                                      n_parties=n_parties, alpha=alpha,
                                                                      n_clusters=n_clusters)
    if nets is None or train_datasets is None or test_dataset is None or global_test_dataset is None:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = init_algorithms(
        algorithm, global_nets, nets, train_datasets, test_dataset, global_test_dataset, args.nk_parties,
        args.epochs, args.rounds, args.learning_rate, args.batch_size, args.weight_decay,
        args.optim_name, device=device, savedir=savedir,
        # kwargs 参数
        mu=args.mu,
        temperature = args.temperature,
        p=args.p,
        split=args.split,
        end=args.end,
        **args.alg_config
    )
    if trainer is not None:
        trainer.start()
    else:
        logging.info("Trainer is None, please check it parameters")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    args = get_args()
    hash_name = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    savedir = os.path.join(args.savedir, hash_name)
    init_logger(savedir, hash_name)
    logging.info(args)
    setup_seed(17)
    train(
        network=args.backbone, datadir=args.datadir, dataset=args.dataset, algorithm=args.alg,
        partition=args.partition, n_parties=args.n_parties, alpha=args.alpha, savedir=savedir,
        n_clusters=args.n_clusters, args=args
    )
