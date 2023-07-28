import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import gzip
import argparse
import tqdm
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
    y_list = []
    i = 0
    """Iterate through all files"""
    for filename in os.listdir(performance_path):

        if model_configs in filename and movement_type in filename and filename.endswith(".pkl.gzip"):
            #print(filename)
            f = gzip.open(performance_path + "/" + filename, "rb")
            data = pickle.load(f)
            print(data)
            x, y = zip(*data)

            x_list.append(x)
            y_list.append(y)
            # plt.plot(x, y)
            # plt.savefig(f"{i}_{model_configs}_{movement_type}")
            # plt.close()
            i = i + 1
        else:
            print(f"No files found for {model_configs} , transformation {movement_type}")
    """Average all episodes y and x values"""

    x_mean = np.mean(x_list, axis=0)
    if movement_type.startswith("rotation"):
        x_mean = x_mean *(180/math.pi)
        x_mean = [map_to_range_180(x) for x in x_mean]
    y_mean = np.mean(y_list, axis=0)
    # fig, ax = plt.subplots()
    # ax.plot(x_mean, y_mean)
    # plt.savefig(f"mean_{model_configs}_{movement_type}")
    return x_mean, y_mean


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
    configs = {
       "(id_1_1)": {"model_name": "SD_DINO", "exp_name": "DINOv1 - stride 4"},
      # "(id_1_1_2)": {"model_name": "SD_DINO", "exp_name": "DINOv1 - stride 8"},
      # "(id_1_2)": {"model_name": "SD_DINO", "exp_name": "DINOv2 - stride 7, layer 11"},
      # "(id_1_2_2)": {"model_name": "SD_DINO", "exp_name": "DINOv2 - stride 7, layer 9"},
       #"(id_1_2_3)": {"model_name": "SD_DINO", "exp_name": "DINOv2 - stride 7, layer 5"},
       # "(id_1_3_2)": {"model_name": "SD_DINO", "exp_name": "SD"},
       #
       # "(id_1_4)": {
       #      "model_name": "SD_DINO",
       #      "descriptor_config_ids": ["(id_1_1)", "(id_1_3_2)"],
       #      "exp_name": "SD + DINOv1 - stride 4"
       #   },
     #    "(id_1_5)": {
     #        "model_name": "SD_DINO",
     #        "descriptor_config_ids": ["(id_1_2)", "(id_1_3_2)"],
     #        "exp_name": "SD + DINOv2 - stride 7, layer 11"
     #    },

     #   "(id_1_6)": {"model_name": "OPEN_CLIP", "exp_name": "OpenCLIP"},
    #    "(id_1_7)": {"model_name": "OPEN_CLIP", "exp_name": "OpenCLIP"},
     #   "(id_2_1)": {"model_name": "SD_DINO", "exp_name": "SD - with captions"},

        # # Alternative metrics
        # "(id_3_1)": {
        #     "model_name": "SD_DINO",
        #     "descriptor_config_ids": ["(id_1_1)"],
        #     "exp_name": "DINOv1 - stride 4, per-head cosine similarity"
        # },
        # "(id_3_2)": {
        #     "model_name": "OPEN_CLIP",
        #     "descriptor_config_ids": ["(id_1_7)"],
        #     "metric": "euclidean",
        #     "exp_name": "OpenCLIP - euclidean similarity"
        # },
        # "(id_3_3)": {
        #     "model_name": "SD_DINO",
        #     "descriptor_config_ids": ["(id_1_3_2)"],
        #     "metric": "euclidean",
        #     "exp_name": "SD - euclidean similarity"
        # },
        "(id_3_4)": {
            "model_name": "SD_DINO",
            "descriptor_config_ids": ["(id_1_1)"],
            "metric": "euclidean",
            "exp_name": "DINOv1 - stride 4, euclidean similarity"
        },
        "(id_1_5_2)": {
            "model_name": "SD_DINO",
            "descriptor_config_ids": ["(id_1_2_2)", "(id_1_3_2)"],
            "exp_name": "SD + DINOv2 - stride 7, layer 9"
        },
        # "(id_3_5)": {
        #     "model_name": "SD_DINO",
        #     "descriptor_config_ids": ["(id_1_2)"],
        #     "metric": "euclidean",
        #     "exp_name": "DINOv2 - stride 7, layer 11, euclidean similarity"
        # },

    }


    for transformation in transformations:
        print(transformation)
        plt.figure()
        for config_id, config in configs.items():
            print(config_id)
            x_mean, y_mean = average_error(model_configs=config_id, movement_type=transformation)

            y_av = movingaverage(y_mean, 1)
            plt.title(f"{transformation}")
            plt.plot(x_mean,
                     y_av, label=config["exp_name"])
            plt.xlim(-.2, 0.2)
            #plt.ylim(0.1, 0.5)
            plt.xlabel("Change in camera position(absolute value)")
            plt.ylabel("Mean error")
            plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.05))
        plt.savefig(f"mean_ALL_MODELS_{transformation}", bbox_inches='tight')
