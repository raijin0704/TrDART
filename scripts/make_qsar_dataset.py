import os

import pandas as pd
import numpy as np

DATASET_DIR = "./dataset/regression/qsar"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00504/qsar_fish_toxicity.csv"
SAVE_NAME_ORIGINAL = "qsar_ori.csv"
SAVE_NAME = "qsar.csv"


HEADER = ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH",
          "NdssC", "MLOGP", "quantitative response"]


def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    df_ori = pd.read_csv(DATASET_URL, sep=";", header=None, names=HEADER)
    df_ori.to_csv(f"{DATASET_DIR}/{SAVE_NAME_ORIGINAL}", index=False)
    # # 変数処理
    df = change_feature(df_ori)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def change_feature(df):
    # SM1_Dz(Z)で分割
    split_col = "SM1_Dz(Z)"
    domains = pd.qcut(df[split_col], 3, labels=['0', '1', '2'])
    # print(domains.value_counts())
    df["SM1_Dz_CAT"] = domains

    return df.drop(columns=[split_col])


def main():
    make_dataset()



if __name__ == "__main__":
    main()
