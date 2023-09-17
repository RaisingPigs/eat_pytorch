import torch.optim.optimizer
from torch import nn
from torchkeras.kerascallbacks import TensorBoardCallback
from torchmetrics import Accuracy
from utils import Config, create_dataloader, show_img, ClassifyNet
from torchkeras import summary, KerasModel


def main():
    train_dl, test_dl = create_dataloader(Config.DATA_DIR, Config.BATCH_SIZE)

    # inputs.shape: [8,3,32,32]
    # targets.shape: [8]
    inputs, targets = next(iter(train_dl))
    show_img(inputs, targets)

    net = ClassifyNet()
    summary(net, inputs)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.LR)
    metrics_dict = {'acc': Accuracy(task='multiclass', num_classes=Config.N_CLASS)}

    # TensorBoard回调函数
    tb = TensorBoardCallback(
        save_dir=Config.LOG_DIR,
        model_name='net',
        log_weight=False,
        log_weight_freq=5,
    )

    kerasModel = KerasModel(
        net,
        loss_fn,
        metrics_dict,
        optimizer
    )
    kerasModel.fit(
        train_dl,
        test_dl,
        Config.EPOCH,
        ckpt_path=Config.CHECKPOINT_FILEPATH,
        patience=Config.PATIENCE,
        callbacks=[tb],  # TensorBoard回调函数
        monitor=Config.MONITOR,
        mode=Config.MODE,
        cpu=True
    )

    # 启动tensorboard, bind_all可以使本机访问虚拟机
    # tensorboard --logdir=./log/cifar10 --port 6006 --bind_all


if __name__ == '__main__':
    main()
