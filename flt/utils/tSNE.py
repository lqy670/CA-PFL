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


# 保证可复现设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(500)
    dataset = CIFAR10Wrapper(
        root="../../data/cifar10",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False,
    )
    loader = data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=4
    )
    model = network.SimpleCNN(num_classes=10,in_channel=3)
    pthfile = '../../exps/2023-05-11-2034-37/models/global_round_50.pth'
    model.load_state_dict(torch.load(pthfile))
    model.cuda()
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for i, (x, y) in enumerate(loader):
            x, y = x.cuda(), y.cuda()
            output = model(x)
            # output1, output2, _ = model(x)
            if i == 0:
                # output1_bank = output1
                # output2_bank = output2
                output_bank = output
                label_bank = y
            else:
                output_bank = torch.cat((output_bank, output))
                # output1_bank = torch.cat((output1_bank, output1))
                label_bank = torch.cat((label_bank, y))
                # output2_bank = torch.cat((output2_bank, output2))
    # feature_bank1 = output1_bank.cpu().numpy()
    # feature_bank2 = output2_bank.cpu().numpy()
    feature_bank = output_bank.cpu().numpy()
    label_bank = label_bank.cpu().numpy()

    tsne = TSNE(2)
    output = tsne.fit_transform(feature_bank)  # feature进行降维，降维至2维表示
    # 带真实值类别
    for i in range(10):  # 对每类的数据画上特定颜色的点
        index = (label_bank == i)
        plt.scatter(output[index, 0], output[index, 1], s=5, cmap=plt.cm.Spectral)
    plt.legend(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    plt.show()
