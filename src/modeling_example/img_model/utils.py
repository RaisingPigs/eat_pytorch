import math
from datetime import datetime

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils import data
import matplotlib.pyplot as plt


class Config():
    # 目录
    BASE_DIR = '../../..'

    DATA_DIR = BASE_DIR + '/data/cifar10'
    LOG_DIR = BASE_DIR + '/log/cifar10'
    OUTPUT_DIR = BASE_DIR + '/output/cifar10'

    CHECKPOINT_DIR = OUTPUT_DIR + '/checkpoint'

    # 训练
    BATCH_SIZE = 64
    LR = 0.01
    EPOCH = 30
    N_CLASS = 10

    # early_stopping
    NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINT_FILEPATH = CHECKPOINT_DIR + f'/{NOW}.pth'
    MONITOR = 'val_acc'
    PATIENCE = 10
    MODE = 'max'


def create_dataloader(filepath, batch_size, ):
    # 获取训练集
    train_ds = datasets.CIFAR10(
        root=filepath,
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    # 获取测试集
    test_ds = datasets.CIFAR10(
        root=filepath,
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )

    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_dl = data.DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_dl, test_dl


def show_img(img_batch: torch.Tensor, label_batch: torch.Tensor):
    plt.figure(figsize=(8, 8))
    size = pow(len(img_batch), 0.5)
    size = math.ceil(size)

    for i, img_tsr in enumerate(img_batch):
        img_np = img_tsr.permute(1, 2, 0).numpy()
        ax = plt.subplot(size, size, i + 1)
        ax.imshow(img_np)
        ax.set_title(f'label={label_batch[i]}')
        ax.set_axis_off()

    plt.show()


class ClassifyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 将网络的整个流程放入Sequential中
        self.net = nn.Sequential(
            # 卷积核为5, 每次卷积高宽会减4, 所以padding=2就能让宽高增加4, 卷积后的tensor高宽仍为32
            nn.Conv2d(3, 32, (5, 5), padding=2),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, (5, 5), padding=2),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (5, 5), padding=2),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)


def plot_metric(history_df, metric):
    train_metrics = history_df["train_" + metric]
    val_metrics = history_df['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)

    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])

    plt.show()
