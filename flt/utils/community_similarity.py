"""
@author: 670
@school: upc
@file: tSNE.py
@date: 2022/11/3 16:27
@desc: 模型特征散点图绘制
"""
import torch
from sklearn.manifold import TSNE  # 这个是绘图关键
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
from flt import network
from flt.dataset.cifar10 import CIFAR10Wrapper
import torchvision
from torch.utils import data
from sklearn.model_selection import train_test_split
from flt import dataset as datastore

# 保证可复现设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def forward_features(model, x):
    # _b = x.shape[0]
    x = model.block_1(x)
    # x1 = x
    x = model.block_2(x)
    # x2 = x
    x = model.block_3(x)
    # x3 = x
    x = model.block_4(x)
    # x4 = x
    x = model.block_5(x)
    x5 = x
    x = model.block_6(x)
    x6 = x
    x = torch.flatten(x, start_dim=1)
    x7 = x
    # return x1.view(_b, -1), x2.view(_b, -1), x3.view(_b, -1), x4.view(_b, -1), x5.view(_b, -1), x6.view(_b, -1)
    return x7


if __name__ == '__main__':
    from flt.dataset import cifar10
    from torch.utils.data import DataLoader
    import pickle
    setup_seed(39)
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


    # dataset = CIFAR10Wrapper(
    #     root="../../data/cifar10",
    #     train=False,
    #     transform=torchvision.transforms.ToTensor(),
    #     download=False,
    # )
    # loader = data.DataLoader(
    #     dataset=dataset,
    #     shuffle=True,
    #     batch_size=4
    # )
    dataloader = DataLoader(dataset=global_test_dataset, batch_size=16, shuffle=True)

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

    model3 = network.SimpleCNN(num_classes=10)
    pthfile3 = '../../exps/2023-06-04-1627-34/models/client7_round_50.pth'
    model3.load_state_dict(torch.load(pthfile3))
    model3.cuda()
    model3.eval()

    model4 = network.SimpleCNN(num_classes=10)
    pthfile4 = '../../exps/2023-06-04-1627-34/models/global_round_50.pth'
    model4.load_state_dict(torch.load(pthfile4))
    model4.cuda()
    model4.eval()

    with torch.no_grad():
        correct, total = 0, 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            output1 = forward_features(model,x)
            output2 = forward_features(model2,x)
            output3 = forward_features(model3,x)
            output4 = forward_features(model4,x)
            # output1, output2, _ = model(x)
            if i == 0:
                # output1_bank = output1
                # output2_bank = output2
                output_bank1 = output1
                output_bank2 = output2
                output_bank3 = output3
                output_bank4 = output4
            else:
                output_bank1 = torch.cat((output_bank1, output1))
                output_bank2 = torch.cat((output_bank2, output2))
                output_bank3 = torch.cat((output_bank3, output3))
                output_bank4 = torch.cat((output_bank4, output4))
                # output1_bank = torch.cat((output1_bank, output1))
                # label_bank = torch.cat((label_bank, y))
                # output2_bank = torch.cat((output2_bank, output2))
    # feature_bank1 = output1_bank.cpu().numpy()
    # feature_bank2 = output2_bank.cpu().numpy()
    sources = [output_bank1,output_bank2,output_bank3,output_bank4]
    angles = torch.zeros(4, 4)
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            angles[i, j] = torch.sum(source1 * source2) / (torch.norm(source1) * torch.norm(source2) + 1e-12)
    print(angles.numpy())



    # label_bank = label_bank.cpu().numpy()
    #
    # tsne = TSNE(n_components=2, init='pca')
    # output = tsne.fit_transform(feature_bank)  # feature进行降维，降维至2维表示
    # output2 = tsne.fit_transform(feature_bank2)  # feature进行降维，降维至2维表示
    # output3 = tsne.fit_transform(feature_bank3)  # feature进行降维，降维至2维表示
    # output4 = tsne.fit_transform(feature_bank4)  # feature进行降维，降维至2维表示
    #
    # fig, axs = plt.subplots(2, 2)  # 创建四格图
    # plt.subplots_adjust(left=0, bottom=0, right=1.0, top=0.93, wspace=0.03, hspace=0.12)
    # ax1 = axs[0, 0]
    # ax2 = axs[0, 1]
    # ax3 = axs[1, 0]
    # ax4 = axs[1, 1]
    # ax1.axis('off')  # 不显示边框
    # ax2.axis('off')  # 不显示边框
    # ax3.axis('off')  # 不显示边框
    # ax4.axis('off')  # 不显示边框
    #
    # ax1.set_title('(1) Client-a Model')
    # ax2.set_title('(2) Client-b Model')
    # ax3.set_title('(3) Community-A model')
    # ax4.set_title('(4) Global Model')
    # for i in range(10):  # 对每类的数据画上特定颜色的点
    #     index = (label_bank == i)
    #     ax1.scatter(output[index, 0], output[index, 1], s=7, alpha=0.8, cmap='Set3_r')
    #     ax2.scatter(output2[index, 0], output2[index, 1], s=7, alpha=0.8, cmap='Set3_r')
    #     ax3.scatter(output3[index, 0], output3[index, 1], s=7, alpha=0.8, cmap='Set3_r')
    #     ax4.scatter(output4[index, 0], output4[index, 1], s=7, alpha=0.8, cmap='Set3_r')
    #
    #
    # plt.axis('off') # 不显示边框
    # # # 带真实值类别
    # # for i in range(10):  # 对每类的数据画上特定颜色的点
    # #     index = (label_bank == i)
    # #     plt.scatter(output[index, 0], output[index, 1], s=10, alpha=0.8,cmap='Set3_r')
    # # ax1.legend(loc='upper left')
    # ax2.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],loc='upper right')
    # # plt.colorbar()  # 显示颜色条
    # plt.show()
