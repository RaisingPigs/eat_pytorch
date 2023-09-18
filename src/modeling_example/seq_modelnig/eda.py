import pandas as pd

from utils import Config

# 读取tsv需要写sep='\t'
train_df = pd.read_csv(Config.TRAIN_FILEPATH, sep="\t", header=None, names=["label", "text"])
test_df = pd.read_csv(Config.TEST_FILEPATH, sep="\t", header=None, names=["label", "text"])

print(train_df.head())
print(test_df.head())
print(train_df.shape)  # (20000, 2)
print(test_df.shape)  # (5000, 2)
