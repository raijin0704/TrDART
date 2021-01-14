import pandas as pd

from tqdm import tqdm

from .experiment_main import experiment_main


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
    bar = tqdm(total=n_domains*n_experiments)
    bar.set_description('Experiments progress')

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
            bar.update(1)

    return results, fit_times