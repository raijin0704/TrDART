import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OrdinalEncoder


WEKA_DIR = './dataset/transfer data/'
PICKLE_DATASETS = ["newsgroups"]

def get_dataset(option_dataset):
    """datasetを取得

    Arguments:
        option_dataset {Object} -- データセットに関するoption

    Raises:
        TypeError: [description]
        TypeError: [description]

    Returns:
        [type] -- [description]
    """
    if option_dataset["source"] == "weka":
        df = _make_dataset_weka(option_dataset)
    elif option_dataset["source"] == "uci":
        df = _make_dataset_uci(option_dataset)
    else:
        raise TypeError('option_dataset["source"]: "{}" is invalid.'\
                            .format(option_dataset["source"]))

    df_reshape = _preprocess_dataset(df, option_dataset)
    return df_reshape


def _make_dataset_weka(option_dataset):
    """先行研究の著者ページに掲載されていたデータを使用

    Arguments:
        option_dataset {Object} -- データセットに関するoption

    Returns:
        pd.DataFrame -- データセット
    """
    dataset_name = option_dataset["dataset_name"]
    try:
        data = arff.loadarff(WEKA_DIR + dataset_name + '1.arff')
        df1 = pd.DataFrame(data[0])
        df1["label"] = 0
        data = arff.loadarff(WEKA_DIR + dataset_name + '2.arff')
        df2 = pd.DataFrame(data[0])
        df2["label"] = 1
        data = arff.loadarff(WEKA_DIR + dataset_name + '3.arff')
        df3 = pd.DataFrame(data[0])
        df3["label"] = 2
        df = pd.concat([df1, df2, df3], sort=False).reset_index(drop=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"{dataset_name} does not exist in WEKA dataset")

    return df


def _make_dataset_uci(option_dataset):
    """uciデータセットを使用

    Args:
        option_dataset {Object} -- データセットに関するoption

    Returns:
        pd.DataFrame -- データセット
    """
    dataset_name = option_dataset["dataset_name"]
    task = option_dataset["task"]
    if dataset_name in PICKLE_DATASETS:
        df = pd.read_pickle(f"./dataset/{task}/{dataset_name}/{dataset_name}.pkl")
        use_col = option_dataset["use_features"]
        df = df.loc[:,:use_col+1]
    else:
        df = pd.read_csv(f"./dataset/{task}/{dataset_name}/{dataset_name}.csv")
    # if task == "regression":
    #     df = _split_domain(df, option_dataset)

    return df


# # TODO 作成(split_featureの値を上書きするように)
# def _split_domain(df, option_dataset):
#     raise TypeError("UCIのregressionは未着手")


def _preprocess_dataset(df, option_dataset):
    """OrdinalEncoding・欠損値除去を行う関数

    Args:
        df (pd.DataFrame): 整形前データフレーム
        option_dataset {Object} -- データセットに関するoption

    Returns:
        pd.DataFrame: 整形後データフレーム
    """
    # 文字列はOrdinalEncoding
    object_cols = df.columns[df.dtypes==object]  # 文字列カラムを取得
    # y_featureとsplit_featureはこの処理を通さないようにする
    drop_cols = [option_dataset["split_feature"], option_dataset["y_feature"]]
    object_cols = [col for col in object_cols if col not in drop_cols]
    df_make = df.copy()
    enc = OrdinalEncoder()
    df_make.loc[:,object_cols] = enc.fit_transform(df_make[object_cols])
    # 欠損値は-9999
    df_make.fillna(-9999, inplace=True)

    return df_make
    