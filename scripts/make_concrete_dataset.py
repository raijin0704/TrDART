import os

import pandas as pd
import numpy as np

DATASET_DIR = "./dataset/regression/concrete"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
SHEET_NAME = "Sheet1"
SAVE_NAME_ORIGINAL = "concrete_ori.csv"
SAVE_NAME = "concrete.csv"


HEADER = ["Cement", "Blast Furnace Slag", "Fly Ash", "Water", 
          "Superplasticizer", "Coarse Aggregate", "Fine Aggregate",
          "Age", "Concrete compressive strength"]


def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    df_ori = pd.read_excel(DATASET_URL, sheet_name=SHEET_NAME, 
                skiprows=[0], names=HEADER)
    df_ori.to_csv(f"{DATASET_DIR}/{SAVE_NAME_ORIGINAL}", index=False)
    # 変数処理
    df = change_feature(df_ori)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def change_feature(df):
    # # Superplasticizerで分割
    # split_col = "Superplasticizer"
    # # split_col=0のデータが多いのでqcutを使わずに分割
    # domains = pd.Series(0,index=df.index)
    # # split_col!=0のデータを分割
    # df_labels12 = df[df["Superplasticizer"]>0]
    # domains[df_labels12.index] = pd.qcut(df_labels12["Superplasticizer"], 2, labels=['1', '2'])
    # print(domains.value_counts())
    # df["Superplasticizer_CAT"] = domains

    split_col = "Water"
    domains = pd.qcut(df[split_col], 3, labels=['0', '1', '2'])
    # print(domains.value_counts())
    df["Water_CAT"] = domains

    return df.drop(columns=[split_col])



def main():
    make_dataset()



if __name__ == "__main__":
    main()
