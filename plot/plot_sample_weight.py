import os

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2

def _plot_sample_weight(sample_weight, group_size, group_name, ymax, iboost, estimator_weight, comment):
    """
    sample_weightを棒グラフに起こす関数
    """
    plt.figure()
    # plt.bar(np.arange(sample_weight.shape[0]), sample_weight)
    plt.ylim(0,ymax)
    before_sum = 0
    for i in range(len(group_size)):
        # 異なるドメインの値をすべて0にする
        mask = np.full(sample_weight.shape[0], True)
        mask[before_sum:group_size[i]+before_sum] = False
        one_sample_weight = sample_weight.copy()
        one_sample_weight[mask] = 0
        
        # plot
        plt.bar(np.arange(sample_weight.shape[0]), one_sample_weight, label=group_name[i])

        before_sum += group_size[i]
    plt.legend(loc='upper left', bbox_to_anchor=(0,1), fontsize=10)
    plt.title("{} iboost={} weight={:3f}".format(comment, iboost, estimator_weight))

    return plt

def save_each_step_plots(sample_weights, group_size, group_name, estimator_weights, dataset_name, save_dir, target_domain_name, comment):
    """
    各イテレーションのsample_weightグラフを保存する関数
    """
    fig_dir = "{}/sample_weight/".format(save_dir)
    os.makedirs(fig_dir, exist_ok=True)
    ymax = sample_weights.max()
    print("------sample_weightのグラフ保存------")
    for iboost in tqdm(range(estimator_weights[estimator_weights>0].shape[0])):
        fig = _plot_sample_weight(sample_weights[iboost,:], group_size, group_name, ymax, iboost, estimator_weights[iboost], comment)
        fig.savefig("{}{}_{:03d}.png".format(fig_dir, target_domain_name, iboost))
        fig.close()
    

def make_movie(dataset_name, save_dir, estimator_weights, target_domain_name):
    """
    イテレーションごとの画像を1つの動画にする
    """
    fig_dir = "{}/sample_weight/".format(save_dir)
    # 画像サイズ取得
    HEGHT, WIDTH, _ = cv2.imread("{}/{}_000.png".format(fig_dir, target_domain_name)).shape

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter('{}/{}_{}_sample_weight.mp4'.format(save_dir, dataset_name, target_domain_name), fourcc, 2.0, (WIDTH, HEGHT))

    # 取り出す画像の数を取得
    image_num = estimator_weights[estimator_weights>0].shape[0]

    for i in range(image_num):
        img = cv2.imread('{}/{}_{:03d}.png'.format(fig_dir, target_domain_name, i))
        img = cv2.resize(img, (640,480))
        video.write(img)

    video.release()



def figure_sample_weight(sample_weights, group_size, group_name, estimator_weights, dataset_name, 
                            save_dir, target_domain_name=None, comment=None):
    """
    全体を実行
    """
    save_each_step_plots(sample_weights, group_size, group_name, estimator_weights, 
                                    dataset_name, save_dir, target_domain_name, comment)
    make_movie(dataset_name, save_dir, estimator_weights, target_domain_name)


if __name__ == "__main__":
    sample_weight = np.full(100, 1/100)
    group_size = np.full(10,10)
    group_name = np.array(["blue", "orange", "green", "red", "perple", "blown", "pink", "grey", "leaf", "sky"])
    fig = _plot_sample_weight(sample_weight, group_size, group_name, 0.1, 1, 0.1)
    fig.show()