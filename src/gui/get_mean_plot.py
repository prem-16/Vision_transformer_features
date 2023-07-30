import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import gzip
import argparse

from matplotlib import patches
from tqdm import tqdm

from src.gui.get_performance import configs, model_categories, model_color_categories

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
    heatmap_error_std = np.std(heatmap_errors_list, axis=0)
    max_point_error_std = np.std(max_point_errors_list, axis=0)

    return x_mean, heatmap_error_mean, max_point_error_mean, heatmap_error_std, max_point_error_std


def main(model_configs, movement_type):
    average_error(model_configs, movement_type)


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def gather_data(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size):
    # Get the configs
    from src.gui.get_performance import configs
    if config_ids is not None:
        # Filter configs by keys that contain a string from configs_ids
        configs = {k: v for k, v in configs.items() if any(x in k for x in config_ids)}

    heatmap_errors_per_transformation = []
    max_point_errors_per_transformation = []
    x_mean_per_transformation = []
    heatmap_std_errors_per_transformation = []
    max_point_std_errors_per_transformation = []

    for transformation in tqdm(transformations):

        x_means = []
        heatmap_errors = []
        max_point_errors = []
        heatmap_std_errors = []
        max_point_std_errors = []

        for config_id, config in configs.items():

            try:
                x_mean, heatmap_error_mean, max_point_error_mean, heatmap_error_std, max_point_error_std = \
                    average_error(model_configs=config_id, movement_type=transformation)

                if apply_log:
                    heatmap_error_mean = np.log(heatmap_error_mean / img_size)
                    max_point_error_mean = np.log(max_point_error_mean / img_size)
                    heatmap_error_std = np.log(heatmap_error_std / img_size) * log_std_scale
                    max_point_error_std = np.log(max_point_error_std / img_size) * log_std_scale
                else:
                    heatmap_error_mean /= img_size
                    max_point_error_mean /= img_size
                    heatmap_error_std = heatmap_error_std * std_scale
                    max_point_error_std = max_point_error_std * std_scale

                if apply_moving_avg:
                    heatmap_error_mean_av = moving_average(heatmap_error_mean, 1)
                    max_point_error_mean_av = moving_average(max_point_error_mean, 1)
                else:
                    heatmap_error_mean_av = heatmap_error_mean
                    max_point_error_mean_av = max_point_error_mean
            except Exception as e:
                print("Failed for config", config_id, "and transformation", transformation)
                raise e

            x_means.append(x_mean)
            heatmap_errors.append(heatmap_error_mean_av)
            max_point_errors.append(max_point_error_mean_av)
            heatmap_std_errors.append(heatmap_error_std)
            max_point_std_errors.append(max_point_error_std)

        x_mean_per_transformation.append(np.array(x_means))
        heatmap_errors_per_transformation.append(np.array(heatmap_errors))
        max_point_errors_per_transformation.append(np.array(max_point_errors))
        heatmap_std_errors_per_transformation.append(np.array(heatmap_std_errors))
        max_point_std_errors_per_transformation.append(np.array(max_point_std_errors))

    return configs, \
        np.array(x_mean_per_transformation), \
        np.array(heatmap_errors_per_transformation), \
        np.array(max_point_errors_per_transformation), \
        np.array(heatmap_std_errors_per_transformation), \
        np.array(max_point_std_errors_per_transformation)


def plot_per_transform(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size):
    configs, x_means_all, heatmap_errors_all, max_point_errors_all, heatmap_std_errors_all, max_point_std_errors_all = \
        gather_data(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size)

    print("Creating plots data...")
    for config, transformation, x_means_per_t, heatmap_errors_per_t, max_point_errors_per_t, \
            heatmap_std_errors_per_t, max_point_std_errors_per_t in tqdm(zip(
        configs, transformations, x_means_all, heatmap_errors_all, max_point_errors_all,
        heatmap_std_errors_all, max_point_std_errors_all
    )):
        for (config_id, config), x_means, heatmap_errors, max_point_errors, \
                heatmap_std_errors, max_point_std_errors in zip(
            configs.items(), x_means_per_t, heatmap_errors_per_t, max_point_errors_per_t,
            heatmap_std_errors_per_t, max_point_std_errors_per_t
        ):
            # PLOT HEATMAP ERROR
            plt.figure(0)
            plt.title(f"Avg error for transformation - {transformation}")
            plt.plot(x_means,
                     heatmap_errors, label=config["exp_name"])
            plt.xlabel("Change in camera position(absolute value)")
            plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}heatmap error")
            plt.fill_between(
                x_means, heatmap_errors - heatmap_std_errors,
                         heatmap_errors + heatmap_std_errors,
                alpha=0.3
            )
            plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.05))
            plt.savefig(f"plots/mean_heatmap_{transformation}", bbox_inches='tight')

            # PLOT MAX POINT ERROR
            plt.figure(1)
            plt.title(f"Avg error for transformation - {transformation}")
            plt.plot(x_means,
                     max_point_errors, label=config["exp_name"])
            plt.xlabel("Change in camera position(absolute value)")
            plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}predicted keypoint distance")
            plt.fill_between(
                x_means, max_point_errors - max_point_std_errors,
                         max_point_errors + max_point_std_errors,
                alpha=0.3
            )
            plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.05))
            plt.savefig(f"plots/mean_max_point_{transformation}", bbox_inches='tight')

        plt.close(fig='all')


def plot_everything_per_transformation(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size):
    configs, _, heatmap_errors_all, max_point_errors_all, heatmap_std_errors_all, max_point_std_errors_all = \
        gather_data(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size)

    max_point_errors = np.mean(max_point_errors_all, axis=-1)
    heatmap_errors = np.mean(heatmap_errors_all, axis=-1)
    # Switch around the axes
    max_point_errors = np.swapaxes(max_point_errors, 0, 1)
    heatmap_errors = np.swapaxes(heatmap_errors, 0, 1)
    width = 5
    height = 5
    transformations = [transformation.replace("_", " ").capitalize() for transformation in transformations]
    exp_names = [config['exp_name'].replace("- ", "\n") for _, config in configs.items()]

    plt.figure(0)
    # Set the height and width of the figure
    plt.figure(figsize=(width, height))
    plt.title(f"Avg point distance error for all transformations")
    # Plot the max point error on the y and the categorical transformations on the x
    for exp_name, errors in zip(exp_names, max_point_errors):
        plt.scatter(
            transformations, errors,
            label=exp_name,
            marker='x'
        )
    plt.xlabel("Transformation")
    plt.xticks(rotation=-45, ha='left')
    plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}error")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"plots/max_point_all_transformations", bbox_inches='tight')
    plt.close()

    plt.figure(1)
    # Set the height and width of the figure
    plt.figure(figsize=(width, height))
    plt.title(f"Avg heatmap error for all transformations")
    # Plot the max point error on the y and the categorical transformations on the x
    for exp_name, errors in zip(exp_names, heatmap_errors):
        plt.scatter(
            transformations, errors,
            label=exp_name,
            marker='x'
        )
    plt.xlabel("Transformation")
    plt.xticks(rotation=-45, ha='left')
    plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}error")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"plots/heatmap_all_transformations", bbox_inches='tight')


def plot_everything(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size):
    configs, _, heatmap_errors_all, max_point_errors_all, heatmap_std_errors_all, max_point_std_errors_all = \
        gather_data(config_ids, transformations, apply_log, apply_moving_avg, std_scale, log_std_scale, img_size)

    max_point_errors = np.mean(max_point_errors_all, axis=-1).mean(axis=0)
    heatmap_errors = np.mean(heatmap_errors_all, axis=-1).mean(axis=0)
    width = 5
    height = 5
    exp_names = [config['exp_name'] for _, config in configs.items()]
    # Gradually colour the bars using a colour map
    # cmap = matplotlib.colormaps.get_cmap('Pastel2')
    # colors = cmap(np.linspace(0, 1, len(exp_names)))
    colors = [model_color_categories[config['category']] for _, config in configs.items()]

    handles = []
    for model_category, color in model_color_categories.items():
        handles.append(patches.Patch(color=color, label=model_category))

    # Order the errors and exp_names by max point error
    max_point_errors, heatmaps_errors, exp_names, colors = \
        zip(*sorted(zip(max_point_errors, heatmap_errors, exp_names, colors)))
    plt.figure(1)
    # Set the height and width of the figure
    plt.figure(figsize=(width, height))
    plt.title(f"Mean keypoint distance over all transformations")
    # Plot the max point error on the y and the categorical transformations on the x
    plt.bar(exp_names, max_point_errors, color=colors)
    plt.xlabel("Model")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )
    plt.legend(handles=handles, loc='upper left')
    plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}keypoint distance")
    plt.ylim(2, 6)
    plt.gcf().set_dpi(300)
    plt.savefig(f"plots/max_point_all", bbox_inches='tight')
    plt.close()

    # Order the errors and exp_names by heatmap error
    heatmap_errors, max_point_errors, exp_names, colors = \
        zip(*sorted(zip(heatmap_errors, max_point_errors, exp_names, colors)))
    plt.figure(1)
    # Set the height and width of the figure
    plt.figure(figsize=(width, height))
    plt.title(f"Mean heatmap error over all transformations")
    # Plot the max point error on the y and the categorical transformations on the x
    plt.bar(exp_names, heatmap_errors, color=colors)
    plt.xlabel("Model")
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )
    plt.legend(handles=handles, loc='upper left')
    plt.ylabel(f"Mean {('(log) ' if APPLY_LOG else '')}heatmap error")
    plt.ylim(4, 6)
    plt.gcf().set_dpi(300)
    plt.savefig(f"plots/heatmap_all", bbox_inches='tight')
    plt.close()


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
    # IMG_SIZE = 512
    IMG_SIZE = 1

    # FILTER CONFIGS BY ID. MAKE SURE TO USE ( )
    # SD Model comparison
    # config_ids = ['(id_1_3_2)', '(id_1_3_3)', '(id_1_3_4)', '(id_3_3)']
    # DINOv1, DINOv2, SD + DINOv1, SD + DINOv2, OpenClip
    # config_ids = ['(id_1_1)', '(id_1_2)', '(id_1_2_3)', '(id_1_4)', '(id_1_5)', '(id_1_6)']
    config_ids = [config_id for config_id in configs.keys() if config_id not in [
        '(id_1_7)', '(id_1_6_2)', '(id_2_2)', '(id_2_3)', '(id_3_2)', '(id_3_3)', '(id_3_4)', '(id_3_5)'
    ]]

    # Plot per transform
    # plot_per_transform(config_ids, transformations, APPLY_LOG, APPLY_MOVING_AVG, STD_SCALE, LOG_STD_SCALE, IMG_SIZE)

    # Aggregate everything for each transform
    # plot_everything_per_transformation(config_ids, transformations, APPLY_LOG, APPLY_MOVING_AVG, STD_SCALE, LOG_STD_SCALE, IMG_SIZE)

    # Aggregate everything
    plot_everything(config_ids, transformations, APPLY_LOG, APPLY_MOVING_AVG, STD_SCALE, LOG_STD_SCALE, IMG_SIZE)
