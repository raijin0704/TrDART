import mlflow


def log_mlflow(options):

    # 開始
    mlflow.set_experiment(options['datasets']["dataset_name"])
    mlflow.start_run()

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