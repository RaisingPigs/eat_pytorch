import pandas as pd
from utils import Config, Tokenizer


def preprocess(data_df: pd.DataFrame, tokenizer: Tokenizer) -> pd.DataFrame:
    text_sr = data_df['text']

    data_df['ids'] = [tokenizer.text2ids(text) for text in text_sr]
    data_df['real_len'] = [tokenizer.get_text_len(text) for text in text_sr]

    return data_df


def main():
    # 读取tsv需要写sep='\t'
    train_df = pd.read_csv(Config.TRAIN_FILEPATH, sep="\t", header=None, names=["label", "text"])
    test_df = pd.read_csv(Config.TEST_FILEPATH, sep="\t", header=None, names=["label", "text"])

    # 第一次从train_df中获取
    # tokenizer = Tokenizer(Config.MAX_LEN, train_df['text'])
    # tokenizer.save(Config.TOKEN_FILEPATH)

    # 第二次直接加载文件
    tokenizer = Tokenizer(Config.MAX_LEN, filepath=Config.TOKEN_FILEPATH)

    train_processed_df = preprocess(train_df, tokenizer)
    test_processed_df = preprocess(test_df, tokenizer)

    train_processed_df.to_csv(Config.TRAIN_PROCESSED_FILEPATH)
    test_processed_df.to_csv(Config.TEST_PROCESSED_FILEPATH)

    train_processed_df.to_pickle(Config.TRAIN_PICKLE_FILEPATH)
    test_processed_df.to_pickle(Config.TEST_PICKLE_FILEPATH)


if __name__ == '__main__':
    main()
