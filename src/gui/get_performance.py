
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

# = image_data["image_rgb"]


def get_error(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))
def get_error_heatmap(heatmap1, heatmap2):
    return np.sqrt(np.average(np.square(heatmap1 - heatmap2)))
load_size = 224

model_manager = ModelGUIManager()
model_manager.update_model("DinoViT")
model_manager.apply_setting("stride", 8)
model_manager.apply_setting("load_size", load_size)
model_manager.apply_setting("layer", 11)
model_manager.apply_setting("facet", "key")
model_manager.apply_setting("threshold", 0.05)
model_manager.apply_setting("model_type", "dino_vits8")
model_manager.apply_setting("log_bin", 0)


# model_manager._image_data_1 = {key: value[0] for key, value in image_data.items()}

DATASET_PATH = ""

# Load the dataset
dataset_data = read_data(datasets_dir="./test_data", dataset_name="data.pkl.gzip")
# Grab the reference image (assumed to be at the first position)
reference_image = dataset_data['image_rgb'][0]

r = cv2.selectROI("select the object", reference_image)
cv2.destroyAllWindows()
plt.imshow(reference_image)
plt.scatter(r[0], r[1], c='r', marker='x')
plt.show()
image1_point = (r[0] / reference_image.shape[1],  r[1] / reference_image.shape[0])
error_list = []
translation_list = []
model_manager._image_data_1 = {key: value[0] for key, value in dataset_data.items()}
# Iterate over remaining images of the sequence
for i, target_image in enumerate(dataset_data['image_rgb'][1:]):
    # Build the cache for the model
    # i.e. set the descriptors, and build the similarities
    model_manager._selected_model.build_cache_from_pkl_gzip("test_data/descriptors/data_descriptor.pkl.gzip", i)
    # Set re-process flag to false
    model_manager._dirty = False
    model_manager._image_data_2 = {key: value[i] for key, value in dataset_data.items()}
    blend_image, _ = model_manager._selected_model.get_heatmap_vis_from_numpy(target_image, image1_point)
    blend_image_ = np.array(blend_image)[:, :, :1]
    pred_index = np.unravel_index(np.argmax(blend_image_, axis=None), blend_image_.shape)[:2]
    pred_index = (pred_index[1], pred_index[0])
    ground_truth_map = model_manager.create_ground_truth_map((r[0], r[1]))
    ground_truth_point = model_manager._transform_points((r[0], r[1]))
    plt.imshow(blend_image)
    plt.scatter(pred_index[0], pred_index[1], c='b', marker='x', label="VIT prediction")
    plt.title("Image Correspondance using DinoViTS8")
    plt.scatter(ground_truth_point[0], ground_truth_point[1], c='r', marker='x', label="ground truth")
    plt.legend()
    plt.show()

    model_manager._dirty = False
    x_translation = model_manager._image_data_2["pose"][0] - model_manager._image_data_1["pose"][0]
    y_translation = model_manager._image_data_2["pose"][1] - model_manager._image_data_1["pose"][1]
    z_translation = model_manager._image_data_2["pose"][2] - model_manager._image_data_1["pose"][2]
    rotation_roll = model_manager._image_data_2["euler_angles"][0] - model_manager._image_data_1["euler_angles"][0]
    print("translation from reference image ", i, " is ", x_translation, y_translation, z_translation)

    # Compute error
    heat_map_pred = model_manager._selected_model.get_heatmap(image1_point)
    heat_map_pred_r= np.resize(heat_map_pred, reference_image.shape[:2])

    error = get_error_heatmap(ground_truth_map, heat_map_pred_r)
    error_list.append(error)
    translation_list.append(rotation_roll)
    print("error for corresponding image from reference image ", i, " is ", error)

list_of_errors = list(zip(translation_list, error_list))
list_of_errors.sort(key=lambda x: x[0])
translation_list, error_list = zip(*list_of_errors)
fig, ax = plt.subplots()
ax.plot(translation_list, error_list)
