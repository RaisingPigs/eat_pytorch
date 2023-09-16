import torch.optim
from torch import nn
from torchmetrics import Accuracy

from utils import Config, get_dataloader, ClassifyNet, Trainer, plot_metric
import pandas as pd


def main():
    x_train_df = pd.read_pickle(Config.X_TRAIN_PICKLE_FILEPATH)
    y_train_df = pd.read_pickle(Config.Y_TRAIN_PICKLE_FILEPATH)
    x_test_df = pd.read_pickle(Config.X_TEST_PICKLE_FILEPATH)
    y_test_df = pd.read_pickle(Config.Y_TEST_PICKLE_FILEPATH)

    train_dl = get_dataloader(x_train_df, y_train_df, Config.BATCH_SIZE)
    test_dl = get_dataloader(x_test_df, y_test_df, Config.BATCH_SIZE)

    for inputs, targets in train_dl:
        # torch.Size([8, 15])
        print(inputs.shape)
        # torch.Size([8, 1])
        print(targets.shape)
        break

    net = ClassifyNet()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=Config.LR)
    metrics_dict = {'acc': Accuracy(task='binary')}
    trainer = Trainer(net, loss_fn, optimizer, metrics_dict)

    history_df = trainer(
        train_dl=train_dl,
        valid_dl=test_dl,
        epochs=Config.EPOCH,
        ckpt_path=Config.CHECKPOINT_FILEPATH,
        patience=Config.PATIENCE,
        monitor=Config.MONITOR,
        mode=Config.MODE,
        log_dir=Config.LOG_DIR
    )

    print(history_df)


def show_():
    history_df = pd.read_csv(Config.LOG_DIR + '/2023-09-16_13-07-39.csv')
    plot_metric(history_df, 'loss')
    plot_metric(history_df, 'acc')


if __name__ == '__main__':
    # main()
    show_()