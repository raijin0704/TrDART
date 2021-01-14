import os

import numpy as np
import pandas as pd


DATASET_DIR = "./dataset/regression/traffic"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
SAVE_NAME_ORIGINAL = "traffic_ori.csv"
SAVE_NAME = "traffic.csv"



def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    df_ori = pd.read_csv(DATASET_URL, parse_dates=["date_time"])
    df_ori.to_csv(f"{DATASET_DIR}/{SAVE_NAME_ORIGINAL}", index=False)
    # 変数処理
    df = change_feature(df_ori)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def change_feature(df):

    # 全ての時刻にholiday情報を入力
    for key, df_one in df.groupby(df["date_time"].dt.date):
        head_holiday = df_one.iloc[0,0]
        df.loc[df_one.index, "holiday"] = head_holiday

    # holidayかどうかフラグ
    df["is_holiday"] = np.where(df["holiday"]=="None", 0, 1)

    # 日時情報
    df["year"] = df["date_time"].dt.year
    df["month"] = df["date_time"].dt.month
    df["day"] = df["date_time"].dt.day
    df["hour"] = df["date_time"].dt.hour
        
        
    delete_cols = ["weather_description", "date_time", "holiday"]
    return df.drop(columns=delete_cols)


def main():
    make_dataset()



if __name__ == "__main__":
    main()
