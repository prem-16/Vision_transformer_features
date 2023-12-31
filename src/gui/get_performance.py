import argparse
import traceback

import matplotlib
import matplotlib.pyplot as plt
import sys
import cv2
from tqdm import tqdm

from src.gui.helpers import get_image_list, read_data, store_data
from src.models.model_gui_manager import ModelGUIManager
import numpy as np
import torch

image_directory = "images/test_images"
import os
import gzip
import pickle


def get_error(point1, point2):
    return np.sqrt(np.average(np.square(point1 - point2)))


def sanitize_point(point, shape):
    point = np.array([max(min(int(point[1]), shape[0] - 1), 0), max(min(int(point[0]), shape[1] - 1), 0)])
    return point


def get_max_error(point, heatmap):
    # Sanitize the point
    point = sanitize_point(point, heatmap.shape)

    # Get the argmax (x, y) of the heatmap
    max_point = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)[:2]

    return np.sqrt((point[0] - max_point[0]) ** 2 + (point[1] - max_point[1]) ** 2)


def get_prop_distance_error(point, heatmap):
    # Sanitize the point to (y, x)
    point = sanitize_point(point, heatmap.shape)

    # Create reference Euclidean distance map
    indices_y, indices_x = np.indices((heatmap.shape[0], heatmap.shape[1]))
    distance_map = np.sqrt(np.sum((
                                          point[:, np.newaxis, np.newaxis] - np.stack((indices_x, indices_y),
                                                                                      axis=0)) ** 2, axis=0
                                  ))

    # Compute the error map
    error_map = distance_map.T * heatmap

    # Sum the error map
    error = np.sum(error_map)

    return error


# model_manager._image_data_1 = {key: value[0] for key, value in image_data.items()}
def get_performance(
        model_name, dataset_name, dataset_path, translation_type: str,
        result_path, descriptor_filenames: list[str], descriptor_paths: list[str],
        output_filename: str, image1_point=None, region=None, metric="cosine", exp_name=None
):
    DATASET_PATH = ""

    # Load the dataset
    model_manager = ModelGUIManager()
    model_manager.update_model(model_name)
    dataset_data = read_data(datasets_dir=dataset_path, dataset_name=dataset_name)
    # descriptor_dir, descriptor_filename = os.path.split(descriptor_dir)
    # _, dataset_name = os.path.split(dataset_path)
    # Grab the reference image (assumed to be at the first position)
    reference_image = dataset_data['image_rgb'][0]

    # Select the object point to find the correspondence.
    if image1_point is None:
        r = cv2.selectROI("Select the object", reference_image)
        cv2.destroyAllWindows()
        image1_point = (r[0] / reference_image.shape[1], r[1] / reference_image.shape[0])
    else:
        # print("Using stored image point", image1_point)
        r = region

    error_list_max_point = []
    error_list_heatmap = []
    translation_list = []
    # for key, value in dataset_data.items():
    #     print(key)
    #     print(len(value))
    model_manager._image_data_1 = {key: value[0] for key, value in dataset_data.items()}

    transformation = {
        "translation_X": [],
        "translation_Y": [],
        "translation_Z": [],
        "rotation_X": [],
        "rotation_Y": [],
        "rotation_Z": [],
    }
    target_patch_size = 100
    model_manager.build_super_cache(pkl_paths=descriptor_paths, target_num_patches=target_patch_size)
    # correspondance name for storing image
    correspondance_name = output_filename.replace(".pkl.gzip", "")
    corr_dir = os.path.join(result_path, correspondance_name)
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)

    plt.imshow(reference_image)
    plt.scatter(r[0], r[1], c='r', marker='x', label="Reference point")
    plt.savefig(os.path.join(corr_dir, f"Reference_image_{exp_name}.png"))
    plt.close()
    # Iterate over remaining images of the sequence
    for i, target_image in enumerate(dataset_data['image_rgb'][0:]):
        # Build the cache for the model
        # i.e. set the descriptors, and build the similarities
        model_manager.build_cache_from_pkl_gzip(descriptor_paths, i, point=image1_point, metric=metric)

        # Set re-process flag to false
        model_manager._dirty = False
        model_manager._image_data_2 = {key: value[i] for key, value in dataset_data.items()}

        # Visualize correspondences and ground truth
        ground_truth_map = model_manager.create_ground_truth_map((r[0], r[1]))
        ground_truth_point = model_manager._transform_points((r[0], r[1]))

        # Compute error
        heat_map_pred = model_manager.selected_model.get_heatmap(image1_point, 0.1)
        assert heat_map_pred.shape == (target_patch_size, target_patch_size), "Heatmap pred shape is not correct"
        # Resize heatmap
        heat_map_pred_r = torch.nn.functional.interpolate(
            torch.tensor(heat_map_pred).reshape(1, 1, heat_map_pred.shape[0], heat_map_pred.shape[1]),
            reference_image.shape[:2], mode='bilinear'
        ).reshape(reference_image.shape[:2])
        heat_map_pred_r = heat_map_pred_r.numpy()
        if i % 5 == 0:
            plt.figure(0)
            # Visualize blend image
            blend_image, _ = model_manager.selected_model.get_heatmap_vis_from_numpy(target_image, image1_point)

            # Compute error

            pred_index = np.unravel_index(np.argmax(heat_map_pred_r, axis=None), heat_map_pred_r.shape)[:2]
            pred_index = (pred_index[1], pred_index[0])
            plt.imshow(blend_image)

            plt.scatter(pred_index[0], pred_index[1], c='b', marker='x', label=exp_name)
            plt.title(f"Image Correspondence")
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='r', marker='x', label="Ground truth")
            plt.legend()
            # print("Image saved in", corr_dir)
            plt.savefig(os.path.join(corr_dir, f"correspondence_{i}_{correspondance_name}.png"))
            plt.close()

        model_manager._dirty = False

        transformation["translation_X"] = (
                model_manager._image_data_2["pose"][0] - model_manager._image_data_1["pose"][0]
        )
        transformation["translation_Y"] = (
                model_manager._image_data_2["pose"][1] - model_manager._image_data_1["pose"][1]
        )
        transformation["translation_Z"] = (
                model_manager._image_data_2["pose"][2] - model_manager._image_data_1["pose"][2]
        )
        transformation["rotation_Z"], transformation["rotation_Y"], transformation["rotation_X"] = [
            a - b
            for a, b in zip(
                model_manager._image_data_2[
                    "euler_angles"],
                model_manager._image_data_1[
                    "euler_angles"]
            )
        ]

        # print(
        #     "translation from reference image to image ", i, " is ", transformation["translation_X"],
        #     transformation["translation_Y"], transformation["translation_Z"]
        # )
        # print(transformation["rotation_Z"])
        # print(transformation["rotation_Y"])
        # print(transformation["rotation_X"])

        # One more norm to prevent any increased mass after resize...
        heat_map_pred_r = heat_map_pred_r / np.sum(heat_map_pred_r)

        # Argmax error get_max_error
        max_point_error = get_max_error(ground_truth_point, heat_map_pred_r)
        # Distance prob error get_prop_distance_error
        heat_map_error = get_prop_distance_error(ground_truth_point, heat_map_pred_r)
        # Append the errors
        error_list_max_point.append(max_point_error)
        error_list_heatmap.append(heat_map_error)
        translation_list.append(transformation[translation_type])

    list_of_errors = list(zip(translation_list, error_list_heatmap, error_list_max_point))
    list_of_errors.sort(key=lambda x: x[0])
    store_data(list_of_errors, "./result", output_filename)
    result_name = output_filename.replace(".pkl.gzip", ".png")
    translation_list, error_list_heatmap, error_list_max_point = zip(*list_of_errors)

    fig, ax = plt.subplots()
    ax.plot(translation_list, error_list_heatmap)

    # Get the x label
    x_label = translation_type
    x_label = translation_type
    # Remove _ from the label
    x_label = x_label.replace("_", " ")
    # Capitalize the first letter
    x_label = x_label.capitalize()

    ax.set(xlabel=x_label, ylabel='Error', title=f"Heatmap prob dist error for {translation_type}")
    plt.savefig(result_path + f"heatmap_result_{correspondance_name}.png")

    fig, ax = plt.subplots()
    ax.plot(translation_list, error_list_max_point)

    # Get the x label
    x_label = translation_type
    x_label = translation_type
    # Remove _ from the label
    x_label = x_label.replace("_", " ")
    # Capitalize the first letter
    x_label = x_label.capitalize()

    ax.set(xlabel=x_label, ylabel='Error', title=f"Euclidean max point error for {translation_type}")
    plt.savefig(result_path + f"max_point_result_{correspondance_name}.png")
    return list_of_errors, image1_point, r


def separate_head_similarity(metric="cosine", head_size=None):
    assert head_size is not None, "Head size must be specified."

    if metric == "cosine":
        metric = torch.nn.CosineSimilarity(dim=3)
    elif metric == "euclidean":
        metric = torch.nn.PairwiseDistance()
    elif metric == "dot":
        metric = torch.dot
    else:
        raise Exception("Unknown metric")

    def metric_func(x, y):
        # We assume x is of shape (b, 1, 1, d)
        # We assume y is of shape (b, 1, t, d)
        # We want to return a tensor of shape (b, 1, t)

        num_splits = x.shape[-1] // head_size
        similarities = torch.zeros((y.shape[0], y.shape[1], y.shape[2], num_splits))

        # For each head in x and y
        for i, (x_head, y_head) in enumerate(zip(
                torch.split(x, num_splits, dim=-1), torch.split(y, num_splits, dim=-1)
        )):
            # Compute similarity
            similarities[..., i] = metric(x_head, y_head)

        return torch.max(similarities, dim=-1, keepdim=False)[0]

    return metric_func


# Define the configs
# Some configs don't have their own descriptors, but rather will be a concatenation
# of other descriptors...
# Be careful!! The first descriptor config id defines the overall concatenated descriptor load_size!

model_categories = ['DINOv1', 'DINOv2', 'SD', 'OpenCLIP', 'SD_COMB']
cmap = matplotlib.colormaps.get_cmap('Pastel2')
colors = cmap(np.linspace(0, 1, len(model_categories)))
model_color_categories = {
    model_name: color for model_name, color in zip(model_categories, colors)
}
config_color = {
    model_name: color for model_name, color in zip(model_categories, colors)
}

configs = {
    "(id_1_1)": {"model_name": "SD_DINO", "exp_name": "DINOv1 - stride 4", "category": "DINOv1"},
    "(id_1_1_2)": {"model_name": "SD_DINO", "exp_name": "DINOv1 - stride 8", "category": "DINOv1"},
    "(id_1_2)": {"model_name": "SD_DINO", "exp_name": "DINOv2 - stride 7, layer 11", "category": "DINOv2"},
    "(id_1_2_2)": {"model_name": "SD_DINO", "exp_name": "DINOv2 - stride 7, layer 9", "category": "DINOv2"},
    "(id_1_2_3)": {"model_name": "SD_DINO", "exp_name": "DINOv2 - stride 7, layer 5", "category": "DINOv2"},
    "(id_1_3_2)": {"model_name": "SD_DINO", "exp_name": "SD - pca", "category": "SD"},

    "(id_1_4)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_1)", "(id_1_3_2)"],
        "exp_name": "SD + DINOv1 - stride 4", "category": "SD_COMB",
    },
    "(id_1_5)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_2)", "(id_1_3_2)"],
        "exp_name": "SD + DINOv2 - stride 7, layer 11", "category": "SD_COMB",
    },

    "(id_1_6)": {"model_name": "OPEN_CLIP", "exp_name": "OpenCLIP", "category": "OpenCLIP"},
    "(id_1_6_2)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_6)", "(id_1_3_4)"],
        "exp_name": "SD + OpenCLIP - s5 only ", "category": "SD_COMB",
    },
    "(id_1_7)": {"model_name": "OPEN_CLIP", "exp_name": "OpenCLIP", "category": "OpenCLIP"},
    "(id_2_2)": {"model_name": "SD_DINO", "exp_name": "SD - with captions and s4 only", "category": "SD"},


    # RUN THESE
    "(id_2_3)": {"model_name": "SD_DINO", "exp_name": "SD - with captions and s5 only", "category": "SD"},
    "(id_1_4_3)": {
        "model_name": "SD + DINOv2 - stride 7, layer 5, s5 only, with captions",
        "descriptor_config_ids": ["(id_1_2_3)", "(id_2_3)"],
        "category": "SD"
    },




    "(id_2_1)": {"model_name": "SD_DINO", "exp_name": "SD - with captions", "category": "SD"},

    "(id_1_5_2)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_2_2)", "(id_1_3_2)"],
        "exp_name": "SD + DINOv2 - stride 7, layer 9",
        "category": "SD_COMB"
    },
    # Alternative metrics
    "(id_3_1)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_1)"],
        "metric": separate_head_similarity(metric="cosine", head_size=6),
        "exp_name": "DINOv1 - stride 4, per-head cosine similarity",
        "category": "DINOv1"
    },
    "(id_3_2)": {
        "model_name": "OPEN_CLIP",
        "descriptor_config_ids": ["(id_1_6)"],
        "metric": "euclidean",
        "exp_name": "OpenCLIP - euclidean similarity",
        "category": "OpenCLIP"
    },
    "(id_3_3)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_3_2)"],
        "metric": "euclidean",
        "exp_name": "SD - euclidean similarity",
        "category": "SD"
    },
    "(id_3_4)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_1)"],
        "metric": "euclidean",
        "exp_name": "DINOv1 - stride 4, euclidean similarity",
        "category": "DINOv1"
    },
    "(id_3_5)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_2)"],
        "metric": "euclidean",
        "exp_name": "DINOv2 - stride 7, layer 11, euclidean similarity",
        "category": "DINOv2"
    },

    # More SD experiments
    "(id_1_3_3)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_3_3)"],
        "exp_name": "SD - s4 only",
        "category": "SD"
    },
    "(id_1_3_4)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_3_4)"],
        "exp_name": "SD - s5 only",
        "category": "SD"
    },

    #SD + DINOv1
    "(id_1_4_2)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_1)", "(id_1_3_4)"],
        "exp_name": "SD + DINOv1 - stride 4, s5 only",
        "category": "SD_COMB"
    },
    # SD + DINOv2
    "(id_1_5_3)": {
        "model_name": "SD_DINO",
        "descriptor_config_ids": ["(id_1_2_3)", "(id_1_3_4)"],
        "exp_name": "SD + DINOv2 - stride 7, layer 5, s5 only",
        "category": "SD_COMB"
    },


}


def get_output_filename(config_id, config, transformation, episode_id):
    return f"{config_id}_result_{config['model_name']}_data_{transformation}_episode_{episode_id}.pkl.gzip"


if __name__ == '__main__':
    # Use argparse to get the arguments
    arg = argparse.ArgumentParser()
    # Model
    arg.add_argument('--model', type=str, default='DinoViT')
    # Data set path
    arg.add_argument('--dataset_path', type=str, default='./test_data/')
    # Descriptor save output path
    arg.add_argument('--descriptor_dir', type=str, default='./test_data/descriptors')
    # Result save output path
    arg.add_argument('--result_path', type=str, default='./result/')
    # Parse the known arguments
    arg.add_argument('--filter_config', type=str, default=None)
    known_args = arg.parse_known_args()[0]

    filter_config = known_args.filter_config
    if filter_config is not None:
        # Delimit the filter config by , and remove any whitespace
        filter_config = [config.strip() for config in filter_config.split(",")]
        # Remove any configs that are not in the filter config
        configs = {
            key: value
            for key, value in configs.items()
            if key in filter_config
        }

    # Parse the arguments
    args = vars(arg.parse_args())
    # Keep track of the errors
    error_list = []
    # Load key points if they exist
    if os.path.exists("./result/key_points.pkl.gzip"):
        f = gzip.open("./result/key_points.pkl.gzip", 'rb')
        image_points = pickle.load(f)
    else:
        image_points = np.full((2, 10), None)

    # Define the transformations
    transformations = [
        "rotation_X",
        "rotation_Y",
        "rotation_Z",
        "translation_X",
        "translation_Y",
        "translation_Z"
    ]

    exceptions_list = []

    # For each configuration i.e. specific model and settings
    for config_id, config in tqdm(configs.items(), position=0, desc="Configurations"):
        # For each transformation
        for transformation in tqdm(transformations, position=1, desc="Transformations"):
            try:
                # print("Generating for config ", config_id, " ", config, " ", transformation)
                for episode_id in tqdm(range(1, 11), position=2, desc="Episodes"):
                    # print("Episode:", episode_id)
                    # Get the dataset file name
                    dataset_file = f"data_{transformation}_episode_{episode_id}.pkl.gzip"
                    # Get the descriptor file name(s)
                    descriptor_configs = config.get("descriptor_config_ids", [config_id])
                    descriptor_filenames = [
                        f"{desc_config_id}_descriptor_{config['model_name']}"
                        f"_data_{transformation}_episode_{episode_id}.pkl.gzip"
                        for desc_config_id in descriptor_configs
                    ]
                    # Get the descriptor file path(s)
                    descriptor_paths = [
                        os.path.join(known_args.descriptor_dir, descriptor_filename)
                        for descriptor_filename in descriptor_filenames
                    ]

                    # Define output file name
                    output_filename = get_output_filename(config_id, config, transformation, episode_id)

                    error, image_point, r = get_performance(
                        model_name=config['model_name'], dataset_name=dataset_file,
                        dataset_path=known_args.dataset_path,
                        translation_type=transformation,
                        result_path=known_args.result_path,
                        image1_point=image_points[0][episode_id - 1],
                        region=image_points[1][episode_id - 1],
                        descriptor_filenames=descriptor_filenames,
                        descriptor_paths=descriptor_paths,
                        output_filename=output_filename,
                        metric=config.get("metric", "cosine"),
                        exp_name=config['exp_name']
                    )
                    image_points[0][episode_id - 1] = image_point
                    image_points[1][episode_id - 1] = r
                    error_list.append(error)

            except:
                print("Exception occurred... Skipping this transformation")
                print(traceback.format_exc())
                exceptions_list.append((config_id, transformation))

        print("Done with config", config_id)

    print("Exceptions list:\n", exceptions_list)

    store_data(image_points, "./result", "key_points.pkl.gzip")
