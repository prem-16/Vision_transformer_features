import argparse
import matplotlib.pyplot as plt
import sys
import cv2
from src.gui.helpers import get_image_list, read_data, store_data
from src.models.model_gui_manager import ModelGUIManager
import numpy as np
import torch

image_directory = "images/test_images"
import os
import gzip
import pickle


def get_error(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))


def get_error_heatmap(heatmap1, heatmap2):
    return np.sqrt(np.average(np.square(heatmap1 - heatmap2)))


# model_manager._image_data_1 = {key: value[0] for key, value in image_data.items()}
def get_performance(
        model_name, dataset_name, dataset_path, translation_type: str,
        result_path, descriptor_filenames: list[str], descriptor_paths: list[str],
        output_filename: str, image1_point=None, region=None
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
        r = cv2.selectROI("select the object", reference_image)
        cv2.destroyAllWindows()
        image1_point = (r[0] / reference_image.shape[1], r[1] / reference_image.shape[0])
    else:
        print("using stored image point", image1_point)
        r = region

    error_list = []
    translation_list = []
    for key, value in dataset_data.items():
        print(key)
        print(len(value))
    model_manager._image_data_1 = {key: value[0] for key, value in dataset_data.items()}

    transformation = {
        "translation_X": [],
        "translation_Y": [],
        "translation_Z": [],
        "rotation_X": [],
        "rotation_Y": [],
        "rotation_Z": [],
    }
    model_manager.build_super_cache(pkl_paths=descriptor_paths)
    # correspondance name for storing image
    correspondance_name = descriptor_filenames[0].replace(".pkl.gzip", "")
    corr_dir = os.path.join(result_path, correspondance_name)
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)

    # Iterate over remaining images of the sequence
    for i, target_image in enumerate(dataset_data['image_rgb'][0:]):
        # Build the cache for the model
        # i.e. set the descriptors, and build the similarities

        model_manager.build_cache_from_pkl_gzip(descriptor_paths, i, point=image1_point)
        # Set re-process flag to false
        model_manager._dirty = False
        model_manager._image_data_2 = {key: value[i] for key, value in dataset_data.items()}

        # Visualize blend image
        blend_image, _ = model_manager.selected_model.get_heatmap_vis_from_numpy(target_image, image1_point)
        blend_image_ = np.array(blend_image)[:, :, :1]
        pred_index = np.unravel_index(np.argmax(blend_image_, axis=None), blend_image_.shape)[:2]
        pred_index = (pred_index[1], pred_index[0])
        plt.imshow(blend_image)

        # Visualize correspondences and ground truth
        ground_truth_map = model_manager.create_ground_truth_map((r[0], r[1]))
        ground_truth_point = model_manager._transform_points((r[0], r[1]))
        if i % 5 == 0:
            plt.scatter(pred_index[0], pred_index[1], c='b', marker='x', label=model_manager.selected_model.NAME)
            plt.title(f"Image Correspondence")
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='r', marker='x', label="ground truth")
            plt.legend()

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

        print(
            "translation from reference image to image ", i, " is ", transformation["translation_X"],
            transformation["translation_Y"], transformation["translation_Z"]
        )
        print(transformation["rotation_Z"])
        print(transformation["rotation_Y"])
        print(transformation["rotation_X"])

        # Compute error
        heat_map_pred = model_manager.selected_model.get_heatmap(image1_point)
        heat_map_pred_r = np.resize(heat_map_pred, reference_image.shape[:2])

        error = get_error_heatmap(ground_truth_map, heat_map_pred_r)
        error_list.append(error)
        translation_list.append(transformation[translation_type])
        print("error for corresponding image from reference image  to image ", i, " is ", error)

    list_of_errors = list(zip(translation_list, error_list))
    list_of_errors.sort(key=lambda x: x[0])
    store_data(list_of_errors, "./result", output_filename)
    result_name = output_filename.replace(".pkl.gzip", ".png")
    translation_list, error_list = zip(*list_of_errors)

    fig, ax = plt.subplots()
    ax.plot(translation_list, error_list)
    plt.savefig(result_path + f"result_{correspondance_name}.png")
    return list_of_errors, image1_point, r


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
    known_args = arg.parse_known_args()[0]

    # Define the configs
    # Some configs don't have their own descriptors, but rather will be a concatenation
    # of other descriptors...
    # Be careful!! The first descriptor config id defines the overall concatenated descriptor load_size!
    configs = {
        "(id_1_1)": {"model_name": "SD_DINO"},
        "(id_1_2)": {"model_name": "SD_DINO"},
        "(id_1_3_2)": {"model_name": "SD_DINO"},
        "(id_1_4)": {
            "model_name": "SD_DINO",
            "descriptor_config_ids": ["(id_1_1)", "(id_1_3_2)"]
        },
        "(id_1_5)": {
            "model_name": "SD_DINO",
            "descriptor_config_ids": ["(id_1_2)", "(id_1_3_2)"]
        },
        "(id_1_6)": {"model_name": "OPEN_CLIP"},
        "(id_1_7)": {"model_name": "OPEN_CLIP"}
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
    transformations = ["rotation_X", "rotation_Y", "rotation_Z", "translation_X", "translation_Y", "translation_Z"]
    # For each transformation
    for transformation in transformations:
        # For each configuration i.e. specific model and settings
        for config_id, config in configs.items():
            print("Generating for config ", config_id, " ", config, " ", transformation)
            for episode_id in range(0, 11):
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
                output_filename = f"{config_id}_result_{config['model_name']}" \
                                  f"_data_{transformation}_episode_{episode_id}.pkl.gzip"

                error, image_point, r = get_performance(
                    model_name=known_args.model, dataset_name=dataset_file,
                    dataset_path=known_args.dataset_path,
                    translation_type=transformation,
                    result_path=known_args.result_path,
                    image1_point=image_points[0][episode_id],
                    region=image_points[1][episode_id],
                    descriptor_filenames=descriptor_filenames,
                    descriptor_paths=descriptor_paths,
                    output_filename=output_filename
                )
                image_points[0][episode_id] = image_point
                image_points[1][episode_id] = r
                error_list.append(error)

    store_data(image_points, "./result", "key_points.pkl.gzip")
