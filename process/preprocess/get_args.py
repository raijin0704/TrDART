import argparse


DATASET_UCI = ["auto_mobile", "autoMPG", "concrete", "housing", "mushroom", 
                "adult", "bank", "banknote", "default", "australian",
                "bike", "traffic", "energy", "qsar"]
SOURCE_UCI = ["weka", "uci"]


def get_uci_args():
    """コマンドライン引数から設定を読み込む

    Returns:
        list -- コマンドライン引数
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="target dataset")
    parser.add_argument("source", help='dataset source ("weka" or "uci")')
    parser.add_argument("-b", "--baseline", default=None, help="baseline folder")
    parser.add_argument("--tune", action="store_true", help="is tuning model params")
    parser.add_argument("-m", "--metrics", default=None, help="set metrics")
    args = parser.parse_args()

    assert args.dataset in DATASET_UCI, \
        'args.dataset: "{}" is invalid.'.format(args.dataset)
    assert args.source in SOURCE_UCI, \
        'args.source: "{}" is invalid.'.format(args.source)

    return args