import os

import pandas as pd


DATASET_DIR = "./dataset/classification/mushroom"
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
SAVE_NAME = "mushroom.csv"

HEADER = ["Attributes", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
          "gill_attachment", "gill_spacing", "gill_size", "gill_color", 
          "stalk_shape", "stalk_root", "stalk_surface_above_ring", 
          "stalk_surface_below_ring", "stalk_color_above_ring", 
          "stalk_color_below_ring", "veil_type", "veil_color", "ring_number", 
          "ring_type", "spore_print_color", "population", "habitat"]


def make_dataset():
    # フォルダが存在しなければ作成
    os.makedirs(DATASET_DIR, exist_ok=True)
    # UCIのwebページからデータセット取得
    df = pd.read_csv(DATASET_URL, header=None, names=HEADER)
    df.to_csv(f"{DATASET_DIR}/{SAVE_NAME}", index=False)


def main():
    make_dataset()



if __name__ == "__main__":
    main()
