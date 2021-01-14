import tempfile
import pathlib
import warnings

import mlflow
import pandas as pd
from tqdm import tqdm

from process.preprocess.get_args import get_uci_args
from process.preprocess.get_dataset import get_dataset
from process.preprocess.get_results import get_results
from process.preprocess.check import check_environment
from process.preprocess.get_set_dir import get_set_dir
from process.preprocess.turn_params import turn_params
from process.preprocess.get_set_options import get_options, set_options
from process.mainprocess.experiment_main import experiment_main
from process.postprocess.main import postprocess


def log_mlflow(options):

    # 開始
    mlflow.set_experiment(options['datasets']["dataset_name"])
    mlflow.start_run(nested=True)

    # タグ
    mlflow.set_tag('experiment_name', options['experiments']["experiment_name"])

    # データセットについて
    mlflow.log_param('source', options['datasets']['source'])
    mlflow.log_param('split_feature', options['datasets']["split_feature"])
    mlflow.log_param('y_feature', options['datasets']['y_feature'])

    # 実験設定
    mlflow.log_param('target_metric', options['experiments']["metrics"])
    mlflow.log_param('n_experiments', options['experiments']["n_experiments"])
    mlflow.log_param('n_targets', options['experiments']["n_targets"])
    mlflow.log_param('source_sampling', options['experiments']["source_sampling"])

    # 手法パラメータ
    mlflow.log_params(options['model_params']['DartForDomainAdaptation'])


def preprocess(methods, n_targets, n_estimators_add, dart_rate, args):
    """前処理

    Returns:
        orderddict -- 実験の設定
        pd.DataFrame -- データセット
        pd.DataFrame -- 引用する結果
        string -- 出力先のディレクトリ
    """

    # 設定の読み込み
    # args = get_uci_args()
    options = get_options(args)

    # target domainのデータ数を変更
    options["experiments"]["n_targets"] = n_targets
    # 実験の試行回数を変更
    options["experiments"]["methods"] = methods
    # params変更
    options["model_params"]["DartForDomainAdaptation"]["n_estimators_add"] = n_estimators_add
    options["model_params"]["DartForDomainAdaptation"]["dart_rate"] = dart_rate
    
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


def mainprocess(options, df):

    # 各パラメータを取り出す
    cols = options["experiments"]["methods"] + ["target_domain"]
    # target domainが明示されていない場合はdomain全てが予測対象
    if options['datasets']['target_domain'] is None:
        domains = df[options['datasets']['split_feature']].unique()
    else:
        domains = options['datasets']['target_domain']
    n_domains = len(domains)
    n_experiments = options['experiments']['n_experiments']

    # tqdm設定
    # bar = tqdm(total=n_domains*n_experiments)
    # bar.set_description('Experiments progress')

    # 実験
    results = pd.DataFrame(columns=cols, index=range(n_domains*n_experiments), dtype=float)
    fit_times = pd.DataFrame(columns=cols, index=range(n_domains*n_experiments), dtype=float)
    for domain_num, target_domain in enumerate(domains):
        for exp_round in range(n_experiments):
            idx = domain_num*n_experiments + exp_round
            # 実験結果の取得
            result, fit_time = experiment_main(options, df, target_domain, exp_round)
            # 手法ごとの実験結果を代入
            for col, val in result.items():
                results.loc[idx, col] = val
            results.loc[idx,cols[-1]] = target_domain
            for col, val in fit_time.items():
                fit_times.loc[idx, col] = val
            fit_times.loc[idx,cols[-1]] = target_domain
            # bar.update(1)

    return results, fit_times




def main():
    args = get_uci_args()
    # 
    mlflow.set_experiment("experiment3")
    mlflow.start_run()
    mlflow.set_tag("datasets", args.dataset)
    
    # 全体の結果を保存
    total_results = pd.DataFrame()
    total_fit_times = pd.DataFrame()

    # この実験は時間がかかるので各条件で10回ずつだけ実行
    # n_experiments = 10

    # 手法は提案手法のみ
    methods = ["DART for DA"]

    # n_targetsの数は10～100（10ずつ）
    n_targets_list = list(range(10, 101, 10))
    # n_estimators_addは0～400（20ずつ）
    n_estimators_add_list = list(range(0, 401, 20))
    # dart_rateは0~1（0.1ずつ）
    dart_rate_list = [i/10.0 for i in range(0, 11)]

    bar_main = tqdm(total=len(n_targets_list) * len(n_estimators_add_list) * len(dart_rate_list))
    bar_main.set_description('All progress')
    
    for n_targets in n_targets_list:
        for n_estimators_add in n_estimators_add_list:
            for dart_rate in dart_rate_list:
                options, df, results_baseline, output_dir = preprocess(
                    methods, n_targets, n_estimators_add, dart_rate, args)
                results, fit_times = mainprocess(options, df)
                postprocess(results, fit_times, results_baseline, output_dir, options)

                results["n_targets"] = n_targets
                results["n_estimators_add"] = n_estimators_add
                results["dart_rate"] = dart_rate
                total_results = pd.concat([total_results, results])

                fit_times["n_targets"] = n_targets
                fit_times["n_estimators_add"] = n_estimators_add
                fit_times["dart_rate"] = dart_rate
                total_fit_times = pd.concat([total_fit_times, fit_times])
                bar_main.update(1)

    with tempfile.TemporaryDirectory() as d:
        all_csv = pathlib.Path(d) / 'all_results.csv'
        mean_csv = pathlib.Path(d) / 'mean_results.csv'
        std_csv = pathlib.Path(d) / 'std_results.csv'
        time_csv = pathlib.Path(d) / 'all_times.csv'
        
        mean_results = total_results.groupby(["n_targets", "n_estimators_add", "dart_rate", "target_domain"]).mean()
        std_results = total_results.groupby(["n_targets", "n_estimators_add", "dart_rate", "target_domain"]).std()

        total_results.to_csv(all_csv, index=False)
        mean_results.to_csv(mean_csv)
        std_results.to_csv(std_csv)
        total_fit_times.to_csv(time_csv, index=False)

        mlflow.log_artifacts(d)
    



if __name__ == "__main__":
    warnings.simplefilter('ignore', UserWarning)
    main()