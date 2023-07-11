import argparse
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageTk
import sys
import cv2
from src.gui.helpers import get_image_list , read_data
from src.models.dino_vit.dino_vit_wrapper import DinoVITWrapper
from src.models.model_gui_manager import ModelGUIManager
import numpy as np
import torch
image_directory = "images/test_images"
image_files, image_dirs = get_image_list(image_directory)
import os
# = image_data["image_rgb"]


def get_error(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))
def get_error_heatmap(heatmap1, heatmap2):
    return np.sqrt(np.average(np.square(heatmap1 - heatmap2)))
load_size = 224



model_manager = ModelGUIManager()
# model_manager.update_model("DinoViT")
# model_manager.apply_setting("stride", 8)
# model_manager.apply_setting("load_size", load_size)
# model_manager.apply_setting("layer", 11)
# model_manager.apply_setting("facet", "key")
# model_manager.apply_setting("threshold", 0.05)
# model_manager.apply_setting("model_type", "dino_vits8")
# model_manager.apply_setting("log_bin", 0)


# model_manager._image_data_1 = {key: value[0] for key, value in image_data.items()}
def get_performance(dataset_name,dataset_path ,translation_type : str,descriptor_dir, result_path):

    DATASET_PATH = ""

    # Load the dataset
    dataset_data = read_data(datasets_dir=dataset_path, dataset_name=dataset_name)
    descriptor_dir, descriptor_filename = os.path.split(descriptor_dir)
    _, dataset_name = os.path.split(dataset_path)
    descriptor_filename = dataset_name.replace("data_", "descriptor_")
    # Grab the reference image (assumed to be at the first position)
    reference_image = dataset_data['image_rgb'][0]
    descriptor_path = os.path.join(descriptor_dir, descriptor_filename)
    # Select the object point to find the correspondence.
    r = cv2.selectROI("select the object", reference_image)
    cv2.destroyAllWindows()
    plt.imshow(reference_image)
    plt.scatter(r[0], r[1], c='r', marker='x')
    plt.show()
    image1_point = (r[0] / reference_image.shape[1],  r[1] / reference_image.shape[0])
    error_list = []
    translation_list = []
    model_manager._image_data_1 = {key: value[0] for key, value in dataset_data.items()}

    transformation = {
        "translation_X": [],
        "translation_Y": [],
        "translation_Z": [],
        "rotation_X": [],
        "rotation_Y": [],
        "rotation_Z": [],
    }

    # Iterate over remaining images of the sequence
    for i, target_image in enumerate(dataset_data['image_rgb'][1:]):
        # Build the cache for the model
        # i.e. set the descriptors, and build the similarities
        model_manager.build_cache_from_pkl_gzip(descriptor_path, i)
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
        if i == 4:
            plt.scatter(pred_index[0], pred_index[1], c='b', marker='x', label=model_manager.selected_model.NAME)
            plt.title(f"Image Correspondence using {model_manager.selected_model.NAME}")
            plt.scatter(ground_truth_point[0], ground_truth_point[1], c='r', marker='x', label="ground truth")
            plt.legend()
            plt.show()
            plt.savefig(os.path.join(result_path, f"correspondence_{i}.png"))

        model_manager._dirty = False

        transformation["translation_X"] = model_manager._image_data_2["pose"][0] - model_manager._image_data_1["pose"][0]
        transformation["translation_Y"]= model_manager._image_data_2["pose"][1] - model_manager._image_data_1["pose"][1]
        transformation["translation_Z"]= model_manager._image_data_2["pose"][2] - model_manager._image_data_1["pose"][2]
        transformation["rotation_X"], transformation["rotation_Y"],transformation["rotation_Z"] = model_manager._image_data_2["euler_angles"][0] - model_manager._image_data_1["euler_angles"][0]
        print("translation from reference image ", i, " is ", transformation["translation_X"], transformation["translation_Y"], transformation["translation_Z"])

        # Compute error
        heat_map_pred = model_manager.selected_model.get_heatmap(image1_point)
        heat_map_pred_r = np.resize(heat_map_pred, reference_image.shape[:2])

        error = get_error_heatmap(ground_truth_map, heat_map_pred_r)
        error_list.append(error)
        translation_list.append(transformation[translation_type])
        print("error for corresponding image from reference image ", i, " is ", error)

    list_of_errors = list(zip(translation_list, error_list))
    list_of_errors.sort(key=lambda x: x[0])
    translation_list, error_list = zip(*list_of_errors)
    fig, ax = plt.subplots()
    ax.plot(translation_list, error_list)
    plt.savefig(result_path)

if __name__ == '__main__':
    # Use argparse to get the arguments
    arg = argparse.ArgumentParser()
    # Model
    arg.add_argument('--model', type=str, default='DinoViT')
    # Data set path
    arg.add_argument('--dataset_path', type=str, default='./test_data/data.pkl.gzip')
    # Descriptor save output path
    arg.add_argument('--descriptor_dir', type=str, default='./test_data/descriptors')
    # Result save output path
    arg.add_argument('--result_path', type=str, default='./result/result.png')
    known_args = arg.parse_known_args()[0]

    # Parse the arguments
    args = vars(arg.parse_args())

    # Get the model manager
    get_performance(dataset_name="data_translation_Z_episode_1.pkl.gzip",dataset_path=known_args.dataset_path,
                    translation_type="translation_Z",descriptor_dir=known_args.descriptor_dir,
                    result_path=known_args.result_path)