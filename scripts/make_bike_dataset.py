import os
from io import BytesIO
import urllib.request
from zipfile import ZipFile

import pandas as pd


DATASET_DIR = "./dataset/regression/bike"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
FILE_NAME = "day.csv"
SAVE_NAME_ORI = "bike_ori.csv"
SAVE_NAME = "bike.csv"



def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    # zipファイルダウンロード
    zip_path = f'{DATASET_DIR}/download_zip.zip'
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    # zipファイルの中身取得
    zip_f = ZipFile(zip_path)
    data_f = zip_f.read(FILE_NAME)
    df_ori = pd.read_csv(BytesIO(data_f))
    df_ori.to_csv(f"{DATASET_DIR}/{SAVE_NAME_ORI}", index=False)

    # 変数処理
    df = change_feature(df_ori)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def change_feature(df):
    # idと日付を削除する
    # cnt = casual + registeredなので2つも削除する
    del_cols = ["instant", "dteday", "casual", "registered"]

    return df.drop(columns=del_cols)


def main():
    make_dataset()



if __name__ == "__main__":
    main()
