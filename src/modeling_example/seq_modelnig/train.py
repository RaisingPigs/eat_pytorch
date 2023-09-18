import pandas as pd
import torch
import torchkeras.kerascallbacks
from torch import nn
from torch.utils.data import DataLoader
from torchkeras import KerasModel
from torchmetrics import Accuracy
from utils import Config, ImdbDataset, Tokenizer
from model import ClassifyNet

train_df = pd.read_pickle(Config.TRAIN_PICKLE_FILEPATH)
test_df = pd.read_pickle(Config.TEST_PICKLE_FILEPATH)
tokenizer = Tokenizer(Config.MAX_LEN, filepath=Config.TOKEN_FILEPATH)

train_ds = ImdbDataset(train_df)
test_ds = ImdbDataset(test_df)

train_dl = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
test_dl = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, drop_last=False)

# inputs.shape: [64, 200]
# targets.shape: [64]
inputs, targets = next(iter(train_dl))
print(inputs.shape)
print(targets.shape)

net = ClassifyNet(tokenizer.size())

tb = torchkeras.kerascallbacks.TensorBoardCallback(
    save_dir=Config.LOG_DIR,
    model_name='net'
)

model = KerasModel(
    net=net,
    # 注意这里损失函数是BCEWithLogitsLoss, targets必须按其规定的方式来
    loss_fn=nn.BCEWithLogitsLoss(),
    optimizer=torch.optim.Adam(net.parameters(), lr=Config.LR),
    metrics_dict={"acc": Accuracy(task='binary')}
)

model.fit(
    train_data=train_dl,
    val_data=test_dl,
    epochs=Config.EPOCH,
    ckpt_path=Config.CHECKPOINT_FILEPATH,
    patience=Config.PATIENCE,
    callbacks=[tb],
    monitor=Config.MONITOR,
    mode=Config.MODE,
    cpu=Config.USE_CPU
)
