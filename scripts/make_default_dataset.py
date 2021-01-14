import os

import pandas as pd
import numpy as np

DATASET_DIR = "./dataset/classification/default"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
SHEET_NAME = "Data"
SAVE_NAME_ORIGINAL = "default_ori.csv"
SAVE_NAME = "default.csv"


HEADER = ["variance of Wavelet Transformed image (continuous)",
          "skewness of Wavelet Transformed image (continuous)",
          "curtosis of Wavelet Transformed image (continuous)",
          "entropy of image (continuous)",
          "class (integer)"]


def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    df_ori = pd.read_excel(DATASET_URL, sheet_name=SHEET_NAME, skiprows=[0])
    df_ori.to_csv(f"{DATASET_DIR}/{SAVE_NAME_ORIGINAL}", index=False)
    # 変数処理
    df = change_feature(df_ori)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def change_feature(df):
    # LIMIT_BALで分割
    split_col = "PAY_3"
    domains = pd.qcut(df[split_col], 3, labels=['0', '1', '2'])
    # print(domains.value_counts())
    df["PAY_3_CAT"] = domains

    return df.drop(columns=[split_col])



def main():
    make_dataset()



if __name__ == "__main__":
    main()
