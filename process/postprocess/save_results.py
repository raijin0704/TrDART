import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns


def save_results(results, fit_times, output_dir, options):
    save_csv(results, output_dir, options)
    plot_result(results, output_dir, options)
    save_time_csv(fit_times, output_dir, options)

    mlflow.log_artifacts(output_dir)




def plot_result(results, output_dir, options):
    # グラフのタイトル
    metrics = options["experiments"]["metrics"]
    title_base = f'{metrics} of {options["datasets"]["dataset_name"]}({options["datasets"]["source"]})'
    # 実行した手法
    methods = options["experiments"]["methods"]
    # 全体の結果plot
    plot_total_result(results, output_dir, title_base, methods)
    # target domainごとのplot
    n_domains = len(results["target_domain"].unique())
    plot_separated_result(results, output_dir, title_base, n_domains, methods)
    # violinplot
    plot_violinplot(results, output_dir, title_base)


def plot_total_result(results, output_dir, title_base, methods):
    
    fig= plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(111)
    df_all_mean = results.mean()[methods]
    df_all_std = results.std()[methods]
    ax1.bar(np.arange(len(df_all_mean)), df_all_mean, yerr=df_all_std,
        tick_label=df_all_mean.index, ecolor="black", width=0.5)
    ax1.set_ylim(bottom=0)
    fig.suptitle(f'{title_base} by total')
    fig.savefig(f'{output_dir}plot_total.png')
    plt.clf()
    plt.close()


def plot_separated_result(results, output_dir, title_base, n_domains, methods):
    
    nrows = math.ceil(n_domains/2)
    fig, ax = plt.subplots(nrows, 2, figsize=(20, 5*nrows), sharey=True, squeeze=False)
    for idx, (domain, df) in enumerate(results.groupby('target_domain')):
        df_mean = df.mean()[methods]
        df_std = df.std()[methods]
        ax[idx//2, idx%2].bar(np.arange(len(df_mean)), df_mean, yerr=df_std,
            tick_label=df_mean.index, ecolor='black', width=0.5)
        ax[idx//2, idx%2].set_ylim(bottom=0)
        ax[idx//2, idx%2].set_title(f'Target domain = {domain}')
    # 奇数個プロットする場合は最後のスペースが余るので空欄にする
    if idx%2==0:
        ax[-1, -1].axis('off')
    fig.suptitle(f'{title_base} by target domain')
    fig.savefig(f'{output_dir}plot_domains.png')
    plt.clf()
    plt.close()


def plot_violinplot(results, output_dir, title_base):
    # データを縦持ちに変更
    df_long = results.set_index("target_domain").stack().reset_index()
    df_long.rename(columns={"level_1":"method", 0:"metric"}, inplace=True)
    # metrics名取得
    metrics = title_base.split(' ')[0]
    # plot
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(1, 1, 1)
    sns.violinplot(x="method", y='metric', hue="target_domain", data=df_long, ax=ax)
    ax.set_ylabel(metrics)
    ax.set_ylim(bottom=0)
    ax.set_title(f'{title_base} violinplot')
    fig.savefig(f'{output_dir}plot_violin.png')
    plt.clf()
    plt.close()


def save_csv(results, output_dir, options):

    # 各イテレーションの結果出力
    results.to_csv(f'{output_dir}result_all.csv', index=False)

    # 実行した手法
    methods = options["experiments"]["methods"]
    
    df_mean = pd.DataFrame(index=methods)
    df_std = pd.DataFrame(index=methods)
    # 全体の結果出力
    df_mean['total'] = results.mean()[methods]
    df_std['total'] = results.std()[methods]   
    # target domainごとの結果出力
    for domain, df in results.groupby('target_domain'):
        df_mean[domain] = df.mean()[methods]
        df_std[domain] = df.std()[methods]
    df_mean.to_csv(f'{output_dir}rmse_mean.csv')
    df_std.to_csv(f'{output_dir}rmse_std.csv')


def save_time_csv(fit_times, output_dir, options):

    # 各イテレーションの学習時間の出力
    fit_times.to_csv(f'{output_dir}fit_time.csv', index=False)
