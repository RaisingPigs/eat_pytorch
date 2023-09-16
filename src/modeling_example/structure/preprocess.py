import pandas as pd
import torch

from utils import Config
from torch.utils.data import DataLoader, TensorDataset


def preprocess(data_df):
    x_df = pd.DataFrame()

    # Pclass
    p_class_df = pd.get_dummies(data_df['Pclass'])
    p_class_df.columns = ['Pclass_' + str(col) for col in p_class_df.columns]
    x_df = pd.concat([x_df, p_class_df], axis=1)

    # sex
    sex_df = pd.get_dummies(data_df['Sex'])
    x_df = pd.concat([x_df, sex_df], axis=1)

    # Age
    x_df['Age'] = data_df['Age'].fillna(0)
    x_df['Age_null'] = pd.isnull(data_df['Age']).astype('int32')

    # SibSp,Parch,Fare
    x_df['SibSp'] = data_df['SibSp']
    x_df['Parch'] = data_df['Parch']
    x_df['Fare'] = data_df['Fare']

    # Cabin
    x_df['Carbin'] = pd.isna(data_df['Cabin']).astype('int32')

    # Embarked
    # dummy_na=True是把nan也作为一列
    embarked_df = pd.get_dummies(data_df['Embarked'], dummy_na=True)
    embarked_df.columns = ['Embarked_' + str(col) for col in embarked_df.columns]
    x_df = pd.concat([x_df, embarked_df], axis=1)

    y_df = data_df[['Survived']]

    return x_df, y_df


def main():
    train_df = pd.read_csv(Config.TRAIN_FILEPATH)
    test_df = pd.read_csv(Config.TEST_FILEPATH)

    x_train_df, y_train_df = preprocess(train_df)
    x_test_df, y_test_df = preprocess(test_df)

    x_train_df.to_csv(Config.X_TRAIN_DF_FILEPATH)
    y_train_df.to_csv(Config.Y_TRAIN_DF_FILEPATH)
    x_test_df.to_csv(Config.X_TEST_DF_FILEPATH)
    y_test_df.to_csv(Config.Y_TEST_DF_FILEPATH)

    x_train_df.to_pickle(Config.X_TRAIN_PICKLE_FILEPATH)
    y_train_df.to_pickle(Config.Y_TRAIN_PICKLE_FILEPATH)
    x_test_df.to_pickle(Config.X_TEST_PICKLE_FILEPATH)
    y_test_df.to_pickle(Config.Y_TEST_PICKLE_FILEPATH)


if __name__ == '__main__':
    main()
