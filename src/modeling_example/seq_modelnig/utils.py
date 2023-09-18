import string
from datetime import datetime

import pandas as pd
import torch
from gensim import corpora
from torch.utils.data import Dataset


class Config():
    # 目录
    BASE_DIR = '../../..'

    DATA_DIR = BASE_DIR + '/data/imdb'
    LOG_DIR = BASE_DIR + '/log/imdb'
    OUTPUT_DIR = BASE_DIR + '/output/imdb'

    CHECKPOINT_DIR = OUTPUT_DIR + '/checkpoint'

    TRAIN_FILEPATH = DATA_DIR + '/train.tsv'
    TEST_FILEPATH = DATA_DIR + '/test.tsv'

    TOKEN_FILEPATH = DATA_DIR + '/token.dict'

    TRAIN_PROCESSED_FILEPATH = DATA_DIR + '/train_processed.csv'
    TEST_PROCESSED_FILEPATH = DATA_DIR + '/test_processed.csv'

    TRAIN_PICKLE_FILEPATH = DATA_DIR + '/train_processed.pickle'
    TEST_PICKLE_FILEPATH = DATA_DIR + '/test_processed.pickle'

    # 训练
    MAX_LEN = 200  # 每个样本保留200个词的长度
    BATCH_SIZE = 64
    LR = 0.01
    EPOCH = 15
    USE_CPU = True

    # early_stopping
    NOW = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    CHECKPOINT_FILEPATH = CHECKPOINT_DIR + f'/{NOW}.pth'
    MONITOR = 'val_acc'
    PATIENCE = 5
    MODE = 'max'


class ImdbDataset(Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __getitem__(self, index):
        inputs = self.data_df['ids'].iloc[index]
        targets = self.data_df['label'].iloc[index]

        inputs = torch.tensor(inputs).long()
        targets = torch.tensor([targets]).float()

        return inputs, targets

    def __len__(self):
        return len(self.data_df)


class Tokenizer():
    def __init__(self, max_len, text_sr: pd.Series = None, filepath=None):
        super().__init__()
        # 将句子拆分为二维数组, 里面的一维数组是一个句子去掉标点符号然后按空格拆分的单词
        self.max_len = max_len
        self.special_tokens = {'<pad>': 0, '<unk>': 1}

        if text_sr is not None:
            self.vocab = corpora.Dictionary((self.text_split(text) for text in text_sr))
            # 过滤数据
            self.vocab.filter_extremes(no_below=5, no_above=5000)

            # 填充词和未知词
            self.vocab.patch_with_special_tokens(self.special_tokens)
        else:
            self.vocab = corpora.Dictionary.load(filepath)

    def text_split(self, text: str):
        """
        将一个句子去掉标点符号, 拆分为单词list
        :param text: 一个英语句子, 如 I like China. 
        :return: ['I', 'like', 'China']
        """

        # string.punctuation: 是所有的标点符号 '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        # str.maketrans(intab,outtab,delchars]) 
        #   intab -- 需要转换的字符组成的字符串。
        #   outtab -- 转换的目标字符组成的字符串。
        #   delchars -- 可选参数，表示要删除的字符组成的字符串。
        translator = str.maketrans('', '', string.punctuation)
        # 把一句英语的标点符号去掉, 然后按照空格拆分为单词数组
        words = text.translate(translator).split(' ')
        return words

    def get_text_len(self, text):
        return len(self.text_split(text))

    def size(self):
        return len(self.vocab.token2id)

    def pad(self, seq, max_length, pad_value=0):
        """序列填充, 超过max_length会被截取"""
        n = len(seq)
        result = seq + [pad_value] * max_length
        return result[:max_length]

    def text2ids(self, text):
        """编码转换"""
        tokens = self.vocab.doc2idx(self.text_split(text))
        tokens = [x if x > 0 else self.special_tokens['<unk>'] for x in tokens]
        result = self.pad(tokens, self.max_len, self.special_tokens['<pad>'])
        return result

    def save(self, filepath):
        self.vocab.save(filepath)
