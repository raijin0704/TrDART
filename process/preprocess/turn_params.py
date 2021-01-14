import os

import optuna
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, \
                             GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


TURNING_TARGET = {
    "GBDT": ["GradientBoostingDecisionTree", "DartForDomainAdaptation"],
    "AdaBoost": ["AdaBoost", "two-stage TrAdaBoost", "MS two-stage TrAdaBoost"]
}

TASK = ["regression", "classification"]


def turn_params(options, output_dir, df, is_plot=True):
    """モデルのチューニングとパラメータの更新
    """
    task = options["datasets"]["task"]
    assert task in TASK, f"{task} is illegal task."
    X, y = get_data(df, options, task)
    for turn_model_name, target_models in TURNING_TARGET.items():
        objective = get_objective(turn_model_name, task, X, y)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=100)
        options = update_params(options, target_models, study.best_trial.params)
        if is_plot:
            plot_turning(study, turn_model_name, output_dir)
    
    return options


def get_data(df_ori, options, task):
    """dfをsampleingし、Xとyに分割
    """
    # sampling
    n_target = options['experiments']['n_targets']
    sample_size = len(df_ori)-n_target
    df = df_ori.sample(sample_size, random_state=0)
    # split
    y_feature = options['datasets']['y_feature']
    domain_feature = options['datasets']['split_feature']
    X = df.drop(columns=[y_feature, domain_feature]).values
    y = df[y_feature].values
    # if task=="regression":
    #     y = df[y_feature].values
    # elif task=="classification":
    #     enc = OneHotEncoder(sparse=False)
    #     y = enc.fit_transform(df[y_feature].values.reshape(-1,1))

    return X, y


def get_objective(model_name, task, X, y):
    """objectiveを取得
    """
    if model_name == "GBDT":
        objective = ObjectiveGBDT(task, X, y)
    elif model_name == "AdaBoost":
        objective = ObjectiveAdaBoost(task, X, y)
    else:
        raise TypeError(f"{model_name}はチューニング未対応")

    return objective


def update_params(options, target_models, best_params):
    """options
    """
    for param_name, param_value in best_params.items():
        for target_model in target_models:
            # AdaBoostの弱学習器パラメータはdictの位置が微妙に違うので調整
            if (target_model in TURNING_TARGET["AdaBoost"]) & (param_name=="max_depth"):
                options["model_params"][target_model]["base_estimator"][param_name] = param_value
            else:
                options["model_params"][target_model][param_name] = param_value

    return options


def plot_turning(study, model_name, output_dir):
    """optunaのvisualization
    """
    fig_dir = f"{output_dir}/turning"
    os.makedirs(fig_dir, exist_ok=True)

    # visualization
    contour_plot = optuna.visualization.plot_contour(study)                 # 等高線
    slice_plot = optuna.visualization.plot_slice(study)                     # パラメータごと
    importance_plot = optuna.visualization.plot_param_importances(study)    # 重要度

    # 保存
    contour_plot.write_image(f'{fig_dir}/{model_name}_contour_plot.png')
    slice_plot.write_image(f'{fig_dir}/{model_name}_slice_plot.png')
    importance_plot.write_image(f'{fig_dir}/{model_name}_importance_plot.png')









class Objective(object):
    def __init__(self, task, X, y):
        self.task = task
        self.X = X
        self.y = y

    def __call__(self, trial):
        self.obj = None
        return self.get_socre(n_jobs=-1, cv=5)
    
    def get_score(self, scoring, n_jobs=-1, cv=5):
        score = cross_val_score(self.obj, self.X, self.y, 
                                scoring=scoring, n_jobs=n_jobs, cv=cv)
        rmse = score.mean()
        return rmse


class ObjectiveGBDT(Objective):
    def __call__(self, trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.5)
        max_depth = trial.suggest_int("max_depth", 2, 32)

        n_estimators = 10000
        validation_fraction = 0.2
        n_iter_no_change = 10

        if self.task == "regression":
            self.obj = GradientBoostingRegressor(
                        n_estimators=n_estimators, learning_rate=learning_rate, 
                        max_depth=max_depth, validation_fraction=validation_fraction, 
                        n_iter_no_change=n_iter_no_change)
            return self.get_score(scoring="neg_root_mean_squared_error", n_jobs=-1, cv=5)
            
        elif self.task == "classification":
            self.obj = GradientBoostingClassifier(
                        n_estimators=n_estimators, learning_rate=learning_rate, 
                        max_depth=max_depth, validation_fraction=validation_fraction, 
                        n_iter_no_change=n_iter_no_change)
            return self.get_score(scoring="neg_log_loss", n_jobs=-1, cv=5)


class ObjectiveAdaBoost(Objective):
    def __call__(self, trial):
        learning_rate = trial.suggest_loguniform("learning_rate", 0.01, 0.5)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        n_estimators = trial.suggest_int('n_estimators', 10, 100, 5)

        loss = "linear"
        random_state = 1

        if self.task == "regression":
            self.obj = AdaBoostRegressor(
                        base_estimator=DecisionTreeRegressor(max_depth=max_depth),
                        n_estimators=n_estimators, learning_rate=learning_rate,
                        loss=loss, random_state=random_state)
            return self.get_score(scoring="neg_root_mean_squared_error", n_jobs=-1, cv=5)

        elif self.task == "classification":
            self.obj = AdaBoostClassifier(
                        base_estimator=DecisionTreeClassifier(max_depth=max_depth),
                        n_estimators=n_estimators, learning_rate=learning_rate,
                        random_state=random_state)
            return self.get_score(scoring="neg_log_loss", n_jobs=-1, cv=5)