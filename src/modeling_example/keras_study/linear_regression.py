import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data as data
from torchkeras import KerasModel, summary
from torchmetrics import MeanAbsoluteError


class Config():
    # 样本数量
    n = 400
    batch_size = 16
    checkpoint_filepath = '../../../output/keras_study/checkpoint.pth'


def create_data(n):
    """
    生成数据集
    """

    # torch.rand([n, 2])生成数据的值为 0~1, 乘10后减5, 则为 -5~5, 有正有负
    # x.shape: [n, 2]
    x = 10 * torch.rand((n, 2)) - 5.0
    w0 = torch.tensor([[2.0], [-3.0]])
    b0 = torch.tensor([[10.0]])

    # @表示矩阵乘法,增加正态扰动
    # 这里的w0和b0假设是真实数据集权重, 这里的y就是真实值
    y = x @ w0 + b0 + torch.normal(0.0, 2.0, size=(n, 1))

    return x, y


def eda(x, y):
    """
    画图
    """
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(121)  # 一行两列, 这是第一个
    ax1.scatter(x[:, 0], y[:, 0], color='r', label='x1 sample')
    ax1.legend()

    plt.title('x1 sample')
    plt.xlabel('x1')
    plt.ylabel('y', rotation=0)

    ax2 = plt.subplot(122)  # 一行两列, 这是第一个
    ax2.scatter(x[:, 1], y[:, 0], color='y', label='x2 sample')
    ax2.legend()

    plt.title('x2 sample')
    plt.xlabel('x2')
    plt.ylabel('y', rotation=0)

    plt.show()


def create_dataloader(x, y, batch_size):
    """
    构建data_loader
    """
    n = x.size(0)
    train_num = int(n * 0.7)  # 训练集的数量, 0.7是比例
    valid_num = n - train_num  # 测试集的数量

    ds = data.TensorDataset(x, y)
    train_ds, valid_ds = data.random_split(ds, (train_num, valid_num))

    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dl = data.DataLoader(valid_ds, batch_size=batch_size, num_workers=2)

    return train_dl, valid_dl


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


def main():
    x, y = create_data(Config.n)
    eda(x, y)
    train_dl, valid_dl = create_dataloader(x, y, Config.batch_size)

    inputs, targets = next(iter(train_dl))
    print(inputs.shape)  # [16, 2]
    print(targets.shape)  # [16, 1]

    net = LinearRegression()
    # 打印显示网络结构和参数
    summary(net, input_data=inputs)

    net = LinearRegression()
    model = KerasModel(
        net=net,
        loss_fn=nn.MSELoss(),
        metrics_dict={"mae": MeanAbsoluteError()},
        optimizer=torch.optim.Adam(net.parameters(), lr=0.01)
    )

    history_df = model.fit(
        train_data=train_dl,
        val_data=valid_dl,
        epochs=100,
        ckpt_path=Config.checkpoint_filepath,
        patience=10,
        monitor='val_loss',
        mode='min'
    )

    print(history_df)


if __name__ == '__main__':
    main()
