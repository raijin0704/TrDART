import os

import pandas as pd
import numpy as np

DATASET_DIR = "./dataset/classification/banknote"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
SAVE_NAME_ORIGINAL = "banknote_ori.csv"
SAVE_NAME = "banknote.csv"


HEADER = ["variance of Wavelet Transformed image (continuous)",
          "skewness of Wavelet Transformed image (continuous)",
          "curtosis of Wavelet Transformed image (continuous)",
          "entropy of image (continuous)",
          "class (integer)"]


def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    df_ori = pd.read_csv(DATASET_URL, header=None, names=HEADER)
    df_ori.to_csv(f"{DATASET_DIR}/{SAVE_NAME_ORIGINAL}", index=False)
    # 変数処理
    df = change_feature(df_ori)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def change_feature(df):
    # curtosis of Wavelet Transformed image (continuous)で分割
    split_col = "curtosis of Wavelet Transformed image (continuous)"
    domains = pd.qcut(df[split_col], 3, labels=['0', '1', '2'])
    # print(domains.value_counts())
    df["curtosis_CAT"] = domains

    return df.drop(columns=[split_col])


def main():
    make_dataset()



if __name__ == "__main__":
    main()
