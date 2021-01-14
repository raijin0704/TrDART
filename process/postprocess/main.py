import mlflow

from .save_results import save_results

def postprocess(results, fit_times, results_baseline, output_dir, options):
    
    # resutls_all = merge_resutls(results, results_baseline)
    save_results(results, fit_times, output_dir, options)
    mlflow.end_run()