import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def plot_error_rmse_by_step(model, test_X, test_y, save_dir, dataset_name, target_name, ex_num):
    rmse_list = []
    for stage2_model in model.models_:
        pred_ = stage2_model.predict(test_X)
        rmse_list.append(mean_squared_error(test_y, pred_)**0.5)
    # stepごとのerror&rmseの推移
    pred_idx = np.array(model.errors_).argmin()
    rmse_min = np.array(rmse_list).argmin()
    # pred_rmse = rmse_list[np.array(model.errors_).argmin()]
    plt.plot(range(len(model.errors_)), np.array(model.errors_)**0.5, label="error")
    plt.plot(range(len(rmse_list)), rmse_list, label="rmse")
    # plt.axvline(pred_idx, color='black', linestyle="dashed")     # hlines
    # plt.scatter(pred_idx, model.errors_[pred_idx]**0.5, color="red")
    # plt.scatter(rmse_min, rmse_list[rmse_min], color="black")
    plt.axhline(rmse_list[pred_idx], color='black', linestyle="dashed", linewidth=0.5)
    plt.axhline(rmse_list[rmse_min], color='red', linestyle="dashed", linewidth=0.5)
    plt.legend()
    plt.title("{} {} num{} : {:.2f}".format(dataset_name, target_name, ex_num, rmse_list[pred_idx]))
    plt.savefig('{}/{}_{}_num{}.png'.format(save_dir, dataset_name, target_name, ex_num))
    plt.close()