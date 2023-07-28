import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import gzip
import argparse

performance_path = "result"



def average_error(model_configs="id_1_1", movement_type="translation_X"):
    x_list = []
    y_list = []
    i = 0
    """Iterate through all files"""
    for filename in os.listdir(performance_path):

        if model_configs in filename and movement_type in filename and filename.endswith(".pkl.gzip"):

            assert filename is not None , f"No files found for {model_configs} , transformation {movement_type}"
            print(filename)
            f = gzip.open(performance_path + "/" + filename, "rb")
            data = pickle.load(f)
            x, y = zip(*data)
            x_list.append(x)
            y_list.append(y)
            # plt.plot(x, y)
            # plt.savefig(f"{i}_{model_configs}_{movement_type}")
            # plt.close()
            i = i + 1
    """Average all episodes y and x values"""

    x_mean = np.mean(x_list, axis=0)
    y_mean = np.mean(y_list, axis=0)
    #fig, ax = plt.subplots()
    #ax.plot(x_mean, y_mean)
   #plt.savefig(f"mean_{model_configs}_{movement_type}")
    return x_mean , y_mean


def main(model_configs, movement_type):
    average_error(model_configs, movement_type)

def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="id_1_1")
    parser.add_argument("--movement", type=str, default="translation_X")
    args = parser.parse_args()
    config_ids = [
                 # "id_1_1",
                #  "id_1_2",
              #    "id_1_4",
                  "id_1_5",
             #     "id_1_6",
             #     "id_1_7"
    ]
    plt.figure()
    for id in config_ids:

        x_mean , y_mean = average_error(model_configs=id, movement_type=args.movement)

        y_av = movingaverage(y_mean,2)
        plt.plot(x_mean,
                 y_av , label =id )
        plt.xlim(-0.2,0.2)
        plt.legend()
    plt.savefig(f"mean_ALL_MODELS_{args.movement}")