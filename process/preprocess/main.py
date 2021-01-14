from .get_args import get_uci_args
from .get_dataset import get_dataset
from .get_results import get_results
from .check import check_environment
from .get_set_dir import get_set_dir
from .turn_params import turn_params
from .get_set_options import get_options, set_options
from .log_mlflow import log_mlflow


# CONFIG_DIR_UCI = './configs/experiments_uci.yml'
# CONFIG_DIR_WEKA = './configs/experiments_weka.yml'


def preprocess():
    """前処理

    Returns:
        orderddict -- 実験の設定
        pd.DataFrame -- データセット
        pd.DataFrame -- 引用する結果
        string -- 出力先のディレクトリ
    """

    # 設定の読み込み
    args = get_uci_args()
    options = get_options(args)
    
    # データセット読み込み
    df = get_dataset(options['datasets'])

    # 引用する結果の読み込み
    # TODO 既に実行していた結果の読み込みに関する部分作成
    if args.baseline:
        results_baseline, options_baseline = get_results()
        check_environment(options, options_baseline)
    else:
        results_baseline = None

    # 出力について設定
    output_dir = get_set_dir(options, args)

    # パラメータチューニングを実行
    if args.tune:
        print("turn model parameters")
        options = turn_params(options, output_dir, df)

    # 実験環境を保存  
    options = set_options(options, output_dir)
    
    # mlflow
    log_mlflow(options)

    return options, df, results_baseline, output_dir

    
