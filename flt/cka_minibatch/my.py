from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.models import resnet18
import torchvision.transforms as transforms

from cka import CKA_Minibatch_Grid
from flt.dataset import cifar10
from sklearn.model_selection import train_test_split
from flt import dataset as datastore


def forward_features(model, x):
    _b = x.shape[0]
    x = model.block_1(x)
    x1 = x
    x = model.block_2(x)
    x2 = x
    x = model.block_3(x)
    x3 = x
    x = model.block_4(x)
    x4 = x
    x = model.block_5(x)
    x5 = x
    x = model.block_6(x)
    x = torch.flatten(x, start_dim=1)
    x6 = x

    return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1), x5.view(_b, -1), x6.view(_b, -1)


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
        for idx, dataidx in dataidx_map.items():
            train_slices[idx], test_slices[idx] = train_test_split(dataidx, train_size=0.75, random_state=31,
                                                                   shuffle=True)
        # logging.info("[Data Cluster] Client 0-2:rotate -- Client 3-5:color -- Client 6-9:origin")
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        for idx, dataidx in train_slices.items():
            if idx >= 0 and idx < 5:  # 旋转90
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((90, 90)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            # elif idx >= 5 and idx < 10:  # 颜色
            else:  # 颜色
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    # transforms.RandomRotation((180, 180)),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            # else:  # 原图
            #     transform = transforms.Compose([
            #         transforms.Resize((32, 32)),
            #         # transforms.RandomRotation((270, 270)),
            #         # GaussianBlur(kernel_size=int(0.1 * 32)),
            #         transforms.ToTensor(),
            #         transforms.Normalize(
            #             mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            #             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            #         )
            #     ])
            train_datasets[idx] = datastore.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                              transform=transform)

        for idx, dataidx in test_slices.items():
            if idx >= 0 and idx < 5:  # 旋转90
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomRotation((90, 90)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            # elif idx >= 5 and idx < 10:  # 颜色
            else:  # 颜色
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                    )
                ])
            # else:  # 高斯滤波
            #     transform = transforms.Compose([
            #         transforms.Resize((32, 32)),
            #         # GaussianBlur(kernel_size=int(0.1 * 32)),
            #         transforms.ToTensor(),
            #         transforms.Normalize(
            #             mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            #             std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            #         )
            #     ])
            test_datasets[idx] = datastore.CIFAR10ALLWrapper(root=datadir, dataidxs=dataidx, download=False,
                                                             transform=transform)

    return train_datasets, test_datasets


if __name__ == '__main__':
    import sys
    from flt import network
    import os
    import pickle
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import torchvision.transforms.functional as VisionF
    from flt.dataset import cifar10

    DATA_ROOT = '../../data/cifar10'
    # dataset = cifar10.CIFAR10ALLWrapper(
    #     root=DATA_ROOT,
    #     # train=True,
    #     download=False,
    # )
    # with open("../../cache/cifar10/practical_non_iid_d_0.5_10.dataidx", "rb") as f:
    #     dataidx_map = pickle.load(f)
    # dataidx_map_count = {k: len(v) for k, v in dataidx_map.items()}
    # print(f"all clientx s samples dataid{dataidx_map_count}")
    # train_datasets, test_dataset = restruce_data_from_dataidx(datadir=DATA_ROOT, dataset="cifar10",
    #                                                           dataidx_map=dataidx_map, n_clusters=3)
    # print("train/test datas and labels for all clients:")
    # for k in range(10):
    #     print(
    #         f"clients {k}: train datas: {train_datasets[k].cls_num_map}, test datas: {test_dataset[k].cls_num_map}")
    # print("---------------------------------------------")
    global_test_dataset = datastore.CIFAR10Wrapper(root=DATA_ROOT, train=False, dataidxs=None, download=False,
                                                   transform=transforms.Compose([
                                                       transforms.Resize((32, 32)),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(
                                                           mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                                           std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                                                       )
                                                   ]))
    batch_size = 16
    num_features = 6

    torch.random.manual_seed(0)
    model = network.SimpleCNN(num_classes=10)
    pthfile = '../../exps/2023-06-04-1627-34/models/client1_round_50.pth'
    model.load_state_dict(torch.load(pthfile))
    model.cuda()
    model.eval()
    model2 = network.SimpleCNN(num_classes=10)
    pthfile2 = '../../exps/2023-06-04-1627-34/models/client2_round_50.pth'
    model2.load_state_dict(torch.load(pthfile2))
    model2.cuda()
    model2.eval()


    cka_logger = CKA_Minibatch_Grid(num_features, num_features)
    with torch.no_grad():
        dataloader = DataLoader(dataset=global_test_dataset, batch_size=16, shuffle=True)
        for images, targets in tqdm(dataloader):
            images = images.cuda()
            features = forward_features(model, images)
            features2 = forward_features(model2, images)
            cka_logger.update(features, features2)
            torch.cuda.empty_cache()
    cka_matrix = cka_logger.compute()

    plt.title('Pretrained SimpleCNN Layer CKA')
    plt.xticks([0, 1, 2, 3,4,5], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6'])
    plt.yticks([0, 1, 2, 3,4,5], ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5', 'Layer 6'])
    print(cka_matrix.numpy())
    plt.imshow(cka_matrix.numpy(), origin='lower', cmap='magma')
    plt.clim(0, 1)
    plt.colorbar()
    plt.savefig('SimpleCNN_cka_12.png')
