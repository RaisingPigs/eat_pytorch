import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from copy import deepcopy


class Config():
    # 目录
    BASE_DIR = '../../..'

    DATA_DIR = BASE_DIR + '/data/titanic'
    LOG_DIR = BASE_DIR + '/log/titanic'
    OUTPUT_DIR = BASE_DIR + '/output/titanic'

    CHECKPOINT_DIR = OUTPUT_DIR + '/checkpoint'

    TRAIN_FILEPATH = DATA_DIR + '/train.csv'
    TEST_FILEPATH = DATA_DIR + '/test.csv'

    X_TRAIN_DF_FILEPATH = DATA_DIR + '/x_train_df.csv'
    Y_TRAIN_DF_FILEPATH = DATA_DIR + '/y_train_df.csv'
    X_TEST_DF_FILEPATH = DATA_DIR + '/x_test_df.csv'
    Y_TEST_DF_FILEPATH = DATA_DIR + '/y_test_df.csv'

    X_TRAIN_PICKLE_FILEPATH = DATA_DIR + '/x_train_df.pickle'
    Y_TRAIN_PICKLE_FILEPATH = DATA_DIR + '/y_train_df.pickle'
    X_TEST_PICKLE_FILEPATH = DATA_DIR + '/x_test_df.pickle'
    Y_TEST_PICKLE_FILEPATH = DATA_DIR + '/y_test_df.pickle'

    # 训练
    BATCH_SIZE = 8
    LR = 0.01
    EPOCH = 20

    # early_stopping
    NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINT_FILEPATH = CHECKPOINT_DIR + f'/{NOW}.pth'
    MONITOR = 'valid_acc'
    PATIENCE = 5
    MODE = 'max'


def get_dataloader(x_df: pd.DataFrame, y_df: pd.DataFrame, batch_size: int):
    ds = TensorDataset(
        torch.tensor(x_df.to_numpy().tolist()).float(),
        torch.tensor(y_df.to_numpy().tolist()).float()
    )

    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=True
    )


class ClassifyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 1)
        )

    def forward(self, inputs):
        return self.net(inputs)


def log(msg):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('\n')
    print('=' * 70 + now)
    print(str(msg))


class StepRunner():
    def __init__(self, net, loss_fn, stage='train', metrics_dict=None, optimizer=None):
        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = metrics_dict
        self.stage = stage
        self.optimizer = optimizer
        self.is_train = stage == 'train'

    def step(self, inputs, targets):
        # loss
        inputs = self.net(inputs)
        loss = self.loss_fn(inputs, targets)

        # backward()
        if self.is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {
            self.stage + "_" + name: metric_fn(inputs, targets).item()
            for name, metric_fn in self.metrics_dict.items()
        }
        return loss.item(), step_metrics

    def train_step(self, inputs, targets):
        self.net.train()  # 训练模式, dropout层发生作用
        return self.step(inputs, targets)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()  # 预测模式, dropout层不发生作用
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == 'train':
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:
    def __init__(self, step_runner):
        self.step_runner = step_runner
        self.stage = step_runner.stage

    def __call__(self, dataloader):
        total_loss = 0
        step = 0

        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        epoch_log = dict()
        for i, batch in loop:
            loss, step_metrics = self.step_runner(*batch)

            total_loss += loss
            step += 1

            if not self.is_last_batch(i + 1, len(dataloader)):
                step_log = dict({self.stage + "_loss": loss}, **step_metrics)
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.step_runner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.step_runner.metrics_dict.items():
                    metric_fn.reset()

        return epoch_log

    def is_last_batch(self, cur_num, batch_len):
        return cur_num == batch_len


class Trainer():
    def __init__(self, net, loss_fn, optimizer, metrics_dict):
        super().__init__()
        self.net = net
        self.train_epoch_runner = self.__create_epoch_runner(net, loss_fn, metrics_dict, 'train', optimizer)
        self.valid_epoch_runner = self.__create_epoch_runner(net, loss_fn, metrics_dict, 'valid')

    def __create_epoch_runner(self, net, loss_fn, metrics_dict, stage='train', optimizer=None):
        step_runner = StepRunner(
            net=net,
            stage=stage,
            loss_fn=loss_fn,
            metrics_dict=deepcopy(metrics_dict),
            optimizer=optimizer
        )
        return EpochRunner(step_runner)

    def __call__(self, train_dl, valid_dl=None,
                 epochs=10, ckpt_path='checkpoint.pth',
                 patience=5, monitor='val_loss', mode='min', log_dir=None):
        history = {}

        for epoch in range(1, epochs + 1):
            log(f'Epoch {epoch} / {epochs}')

            metrics_log = self.train_epoch_runner(train_dl)

            if valid_dl:
                val_metrics = self.valid_epoch_runner(valid_dl)
                metrics_log.update(val_metrics)

            metrics_log["epoch"] = epoch
            for name, metric in metrics_log.items():
                history[name] = history.get(name, []) + [metric]

            # early-stopping -------------------------------------------------
            arr_scores = history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores) - 1:
                torch.save(self.net.state_dict(), ckpt_path)
                print(f"<<<<<< reach best {monitor} : {arr_scores[best_score_idx]} >>>>>>")
            if len(arr_scores) - best_score_idx > patience:
                print(f'<<<<<< {monitor} without improvement in {patience} epoch, early stopping >>>>>>')
                break
            self.net.load_state_dict(torch.load(ckpt_path))

        history_df = pd.DataFrame(history)
        if log_dir:
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_path = os.path.join(log_dir, now + ".csv")
            history_df.round(2).to_csv(log_path, index=False)

        return history_df


def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_" + metric]
    valid_metrics = dfhistory['valid_' + metric]
    
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, valid_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()
