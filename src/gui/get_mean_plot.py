import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import gzip
import argparse
from tqdm import tqdm

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
            x, heatmap_errors, max_point_errors = zip(*data)
            x_list.append(x)
            heatmap_errors_list.append(heatmap_errors)
            max_point_errors_list.append(max_point_errors)
            i += 1

    x_mean = np.mean(x_list, axis=0)
    if movement_type.startswith("rotation"):
        x_mean = x_mean * (180 / math.pi)
        x_mean = [map_to_range_180(x) for x in x_mean]

    heatmap_error_mean = np.mean(heatmap_errors_list, axis=0)
    max_point_error_mean = np.mean(max_point_errors_list, axis=0)
    heatmap_error_std = np.std(heatmap_error_mean , axis=0)
    max_point_error_std = np.std(max_point_errors_list ,axis=0)

    return x_mean, heatmap_error_mean, max_point_error_mean, heatmap_error_std, max_point_error_std



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

    # VARIABLES
    APPLY_LOG = True
    APPLY_MOVING_AVG = True
    STD_SCALE = 0.5
    LOG_STD_SCALE = 0.1

    # FILTER CONFIGS BY ID. MAKE SURE TO USE ( )
    # SD Model comparison
    # config_ids = ['(id_1_3_2)', '(id_1_3_3)', '(id_1_3_4)', '(id_3_3)']
    # DINOv1, DINOv2, SD + DINOv1, SD + DINOv2, OpenClip
    config_ids = ['(id_1_1)', '(id_1_2)', '(id_1_2_3)', '(id_1_4)', '(id_1_5)', '(id_1_6)']
    # Filter configs by keys that contain a string from configs_ids
    configs = {k: v for k, v in configs.items() if any(x in k for x in config_ids)}

    for transformation in tqdm(transformations):
        for config_id, config in configs.items():

            x_mean, heatmap_error_mean, max_point_error_mean, heatmap_error_std, max_point_error_std = \
                average_error(model_configs=config_id, movement_type=transformation)

            if APPLY_LOG:
                heatmap_error_mean = np.log(heatmap_error_mean)
                max_point_error_mean = np.log(max_point_error_mean)
                heatmap_error_std = np.log(heatmap_error_std) * LOG_STD_SCALE
                max_point_error_std = np.log(max_point_error_std) * LOG_STD_SCALE
            else:
                heatmap_error_std = heatmap_error_std * STD_SCALE
                max_point_error_std = max_point_error_std * STD_SCALE

            if APPLY_MOVING_AVG:
                heatmap_error_mean_av = movingaverage(heatmap_error_mean, 1)
                max_point_error_mean_av = movingaverage(max_point_error_mean, 1)
            else:
                heatmap_error_mean_av = heatmap_error_mean
                max_point_error_mean_av = max_point_error_mean

            # PLOT HEATMAP ERROR
            plt.figure(0)
            plt.title(f"Avg error for transformation - {transformation}")
            plt.plot(x_mean,
                     heatmap_error_mean_av, label=config["exp_name"])
            plt.xlabel("Change in camera position(absolute value)")
            plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}heatmap error")
            plt.fill_between(
                x_mean, heatmap_error_mean_av - heatmap_error_std,
                heatmap_error_mean_av + heatmap_error_std,
                alpha=0.3
            )
            plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.05))
            plt.savefig(f"plots/mean_heatmap_{transformation}", bbox_inches='tight')

            # PLOT MAX POINT ERROR
            plt.figure(1)
            plt.title(f"Avg error for transformation - {transformation}")
            plt.plot(x_mean,
                     max_point_error_mean_av, label=config["exp_name"])
            plt.xlabel("Change in camera position(absolute value)")
            plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}predicted keypoint distance")
            plt.fill_between(
                x_mean, max_point_error_mean_av - max_point_error_std,
                max_point_error_mean_av + max_point_error_std,
                alpha=0.3
            )
            plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.05))
            plt.savefig(f"plots/mean_max_point_{transformation}", bbox_inches='tight')

        plt.close(fig='all')
    # plt.savefig(f"plots/mean_ALL_MODELS_{args.movement}")
