import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import gzip
import argparse
import tqdm

from src.gui.get_performance import configs

performance_path = "result"
import math


def map_to_range_180(value):
    while value > 180:
        value -= 360
    while value <= -180:
        value += 360
    return value


def average_error(model_configs="id_1_1", movement_type="translation_X"):
    x_list = []
    heatmap_errors_list = []
    max_point_errors_list = []
    i = 0
    """Iterate through all files"""
    for filename in os.listdir(performance_path + "/"):

        if model_configs in filename and movement_type in filename and filename.endswith(".pkl.gzip"):
            # print(filename)
            f = gzip.open(performance_path + "/" + filename, "rb")
            data = pickle.load(f)
            print(data)
            x, heatmap_errors, max_point_errors = zip(*data)

            x_list.append(x)
            heatmap_errors_list.append(heatmap_errors)
            max_point_errors_list.append(max_point_errors)
            # plt.plot(x, y)
            # plt.savefig(f"{i}_{model_configs}_{movement_type}")
            # plt.close()
            i += 1

    print("Number of episodes ", i)
    """Average all episodes y and x values"""

    x_mean = np.mean(x_list, axis=0)
    if movement_type.startswith("rotation"):
        x_mean = x_mean * (180 / math.pi)
        x_mean = [map_to_range_180(x) for x in x_mean]
    heatmap_error_mean = np.mean(heatmap_errors_list, axis=0)
    max_point_error_mean = np.mean(max_point_errors_list, axis=0)
    # fig, ax = plt.subplots()
    # ax.plot(x_mean, y_mean)
    # plt.savefig(f"mean_{model_configs}_{movement_type}")
    return x_mean, heatmap_error_mean, max_point_error_mean


def main(model_configs, movement_type):
    average_error(model_configs, movement_type)


def movingaverage(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="id_1_1")
    parser.add_argument("--movement", type=str, default="translation_X")
    args = parser.parse_args()

    transformations = [
        "rotation_X",
        "rotation_Y",
        "rotation_Z",
        "translation_X",
        "translation_Y",
        "translation_Z"
    ]

    # Config ids to plot
    # config_ids = ['(id_1_1)', '(id_1_2)', '(id_1_4)', '(id_1_5), (id_1_6)']
    # config_ids = ['id_1_6', 'id_3_2']
    # config_ids = ['id_1_1', 'id_1_1_2']
    # Filter configs by keys that contain a string from configs_ids
    # configs = {k: v for k, v in configs.items() if any(x in k for x in config_ids)}


    for transformation in transformations:
        print(transformation)
        plt.figure()
        for config_id, config in configs.items():
            print(config_id)
            x_mean, heatmap_error_mean, max_point_error_mean = \
                average_error(model_configs=config_id, movement_type=transformation)

            y_av = movingaverage(np.log(heatmap_error_mean), 1)
            plt.title(f"{transformation}")
            plt.plot(x_mean,
                     y_av, label=config["exp_name"])
            plt.xlim(-.2, 0.2)
            # plt.ylim(0.1, 0.5)
            plt.xlabel("Change in camera position(absolute value)")
            plt.ylabel("Mean error")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1))
        plt.savefig(f"plots/mean_ALL_MODELS_{transformation}", bbox_inches='tight')

        x_mean, heatmap_error_mean, max_point_error_mean = \
            average_error(model_configs=config_id, movement_type=args.movement)

        y_av = movingaverage(np.log(heatmap_error_mean), 2)
        plt.plot(x_mean,
                 y_av, label=id)
        plt.xlim(-0.2, 0.2)
        plt.legend()

    # plt.savefig(f"plots/mean_ALL_MODELS_{args.movement}")
