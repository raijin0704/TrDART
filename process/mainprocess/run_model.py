import time

from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import OneHotEncoder
import mlflow
import numpy as np

from .models import AdaBoost, GradientBoostingDecisionTree, TrAdaBoost, \
                    TwoStageTrAdaBoostR2, MultipleSourceTwoStageTrAdaBoostR2, \
                    Trbagg, TransferStacking,\
                    DartForDomainAdaptation


# DART_MODELS = ["DART for DA", "DART for DA (worse)", "DART for DA (latest)"]
DART_MODELS = ["TrDART", "only Dropout", "target DART"]
RETURN_PROB_METRICS = ["auc", "f1"]


def run_model(X_train, y_train, 
                X_train_target, y_train_target, 
                X_test, y_test, domain_col, options, 
                model_name, target_domain, exp_round):
    
    # いろんなモデルを実行
    if model_name == "target AdaBoost":
        model = AdaBoost(options, model_name)
        start = time.time()
        model.fit(X_train_target, y_train_target)
        fit_time = time.time() - start

        

    elif model_name == "all AdaBoost":
        model = AdaBoost(options, model_name)
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start
        

    elif model_name == "TrAdaBoost":
        model = TrAdaBoost(options, model_name)
        start = time.time()
        model.fit(X_train, y_train, domain_col, target_domain)
        fit_time = time.time() - start
        

    elif model_name == "two-stage TrAdaBoost":
        model = TwoStageTrAdaBoostR2(options, model_name)
        start = time.time()
        model.fit(X_train, y_train, domain_col, target_domain)
        fit_time = time.time() - start
        

    elif model_name == "MS two-stage TrAdaBoost":
        model = TwoStageTrAdaBoostR2(options, model_name)
        start = time.time()
        model.fit(X_train, y_train, domain_col, target_domain)
        fit_time = time.time() - start
    

    elif model_name == "Trbagg":
        model = Trbagg(options, model_name)
        start = time.time()
        model.fit(X_train, y_train, domain_col, target_domain, X_test)
        fit_time = time.time() - start
        

    elif model_name == 'TransferStacking':
        # expert_modelの学習
        expert_model_name = options['model_params']['TransferStacking']['expert_model']
        if expert_model_name=='AdaBoost':
            X_train_source = X_train[domain_col!=target_domain]
            y_train_source = y_train[domain_col!=target_domain]
            expert_model = AdaBoost(options, expert_model_name)
            expert_model.fit(X_train_source, y_train_source)
        else:
            msg = 'Expert model of TransferStacking "{}" is invalid'.format(
                expert_model_name
            )
            raise TypeError(msg)
        model = TransferStacking(options, model_name, expert_model.model)
        start = time.time()
        model.fit(X_train_target, y_train_target)
        fit_time = time.time() - start
        

    elif model_name == "target GBDT":
        model = GradientBoostingDecisionTree(options, model_name)
        model.model.set_params(n_iter_no_change =None, n_estimators=20)
        start = time.time()
        model.fit(X_train_target, y_train_target)
        fit_time = time.time() - start
        

    elif model_name == "all GBDT":
        model = GradientBoostingDecisionTree(options, model_name)
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start
        
    
    elif model_name in DART_MODELS:
        X_train_source = X_train[domain_col!=target_domain]
        y_train_source = y_train[domain_col!=target_domain]
        model = DartForDomainAdaptation(options, model_name)
        model.fit_source(X_train_source, y_train_source)
        n_tree_source = model.model.n_estimators_
        start = time.time()
        model.fit(X_train_target, y_train_target)
        fit_time = time.time() - start
        

    metrics = options["experiments"]["metrics"]
    if metrics in RETURN_PROB_METRICS:
        y_pred = model.predict_proba(X_test)
        test_classes_ = np.searchsorted(model.label_class, y_test)
        le = OneHotEncoder()
        y_test_ = le.fit_transform(test_classes_.reshape(-1,1)).toarray()
        metric_value = return_metrics(y_test_, y_pred, metrics)
    else:
        y_pred = model.predict(X_test)
        metric_value = return_metrics(y_test, y_pred, metrics)

    # mlflow
    if model_name == "TrDART":
        mlflow.log_metric(f'{metrics}_{target_domain}', metric_value, step=exp_round)
        mlflow.log_metric(f'source_estimators_{target_domain}', n_tree_source, step=exp_round)
        mlflow.log_metric(f"drop_rate_{target_domain}", model.model.drop_rate, step=exp_round)

    
    return metric_value, fit_time



def return_metrics(y_true, y_pred, metrics):
    if metrics=="MSE":
        return mean_squared_error(y_true, y_pred)
    elif metrics=="RMSE":
        return mean_squared_error(y_true, y_pred)**0.5
    elif metrics=="accuracy":
        return accuracy_score(y_true, y_pred)
    elif metrics=="error rate":
        return 1 - accuracy_score(y_true, y_pred)
    elif metrics=="auc":
        return roc_auc_score(y_true, y_pred)
    elif metrics=="f1":
        return f1_score(y_true, y_pred)
    else:
        raise TypeError(f"{metrics} is illegal metrics.")