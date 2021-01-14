from .split_data import split_data
from .run_model import run_model

def experiment_main(options, df, target_domain, exp_round):

    X_train, y_train, X_train_target, y_train_target, X_test, y_test, domain_col = \
        split_data(options, df, target_domain, exp_round)


    results = {}
    fit_times = {}

    for model_name in options["experiments"]["methods"]:
        result, fit_time = run_model(
            X_train, y_train, X_train_target, y_train_target, X_test, y_test, 
            domain_col, options, model_name, target_domain, exp_round
        )
        results[model_name] = result
        fit_times[model_name] = fit_time

    return results, fit_times