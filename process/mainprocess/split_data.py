import pandas as pd
import numpy as np


def split_data(options, df, target_domain, exp_round):

    n_target = options['experiments']['n_targets']
    y_feature = options['datasets']['y_feature']
    split_col =  options['datasets']['split_feature']

    # データの分割
    # targetデータはデータ数を小さくするためランダムサンプリング
    target_df = df.query("{} == '{}'".format(split_col, target_domain))\
                    .sample(n=n_target, random_state=exp_round)
    # y_kinds = 0
    # max_y_kinds = len(df[y_feature].unique())
    # seed = exp_round + 1
    # while y_kinds < max_y_kinds:
    #     target_df = df.query("{} == '{}'".format(split_col, target_domain))\
    #                 .sample(n=n_target, random_state=seed)
    #     y_kinds = len(target_df[y_feature].unique())
    #     seed = (seed * exp_round + 2) % (2**32)
    target_X_train = target_df.drop(columns=y_feature).drop(columns=split_col).values
    domain_col_target = target_df[split_col].values
    target_Y_train = target_df[y_feature].values
    # サンプリングされなかったデータをtestデータとする
    target_df_test = df.query("{} == '{}'".format(split_col, target_domain))\
                     .drop(index=target_df.index)
    target_X_test = target_df_test.drop(columns=y_feature).drop(columns=split_col).values
    target_Y_test = target_df_test[y_feature].values
    
    if options['experiments']['source_sampling']:
        n_source = options['experiments']['source_sampling']
        df_sample = df.query(f"{split_col} != @target_domain")\
                        .sample(n=n_source, random_state=exp_round)
        source_df = df_sample.query(f"{split_col} != @target_domain").sort_values(split_col)
    else:
        source_df = df.query(f"{split_col} != @target_domain").sort_values(split_col)
    
    source_X = source_df.drop(columns=y_feature).drop(columns=split_col).values
    source_Y = source_df[y_feature].values
    domain_col_source = source_df[split_col].values

    X = np.concatenate([source_X, target_X_train])
    y = np.concatenate([source_Y, target_Y_train])
    domain_col = np.concatenate([domain_col_source, domain_col_target])


    # sourceとtargetの学習用データが全クラスあるかどうかチェック
    if options["datasets"]["task"] == "classification":
        if len(set(source_Y)) != len(set(target_Y_train)):
            # print("targetの学習用データのラベルが不足しています。seedを変更して再サンプリングします")
            X, y, target_X_train, target_Y_train, target_X_test, target_Y_test, \
                domain_col = split_data(options, df, target_domain, exp_round*100)


    return X, y, target_X_train, target_Y_train, target_X_test, target_Y_test, domain_col
        