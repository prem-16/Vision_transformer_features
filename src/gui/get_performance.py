
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageTk
import sys
import cv2
from src.gui.helpers import get_image_list , read_data
from src.models.model_gui_manager import ModelGUIManager
from src.models.model_wrapper_list import MODEL_DICT
import numpy as np
import torch
image_directory = "images/test_images"
image_files, image_dirs = get_image_list(image_directory)

image_data = read_data(datasets_dir="./test_data")
# = image_data["image_rgb"]


def get_error(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))
load_size = 224

model_manager = ModelGUIManager()
model_manager = ModelGUIManager()
model_manager.update_model("DinoViT")
model_manager.apply_setting("stride", 8)
model_manager.apply_setting("load_size", load_size)
model_manager.apply_setting("layer", 11)
model_manager.apply_setting("facet", "key")
model_manager.apply_setting("threshold", 0.05)
model_manager.apply_setting("model_type", "dino_vits8")

model_manager.image_dir_1 = image_dirs[0]

model_manager._image_data_1 = {key: value[0] for key, value in image_data.items()}
image1 = cv2.imread(image_dirs[0])

r = cv2.selectROI("select the object", image1)
cv2.destroyAllWindows()
plt.imshow(image1)
plt.scatter(r[0],r[1], c='r', marker='x')
plt.show()
image1_point = (r[0]/image1.shape[1],  r[1]/image1.shape[0])
for i, image_dir in enumerate(image_dirs[1:]):
    model_manager.image_dir_2 = image_dir
    model_manager._image_data_2 = {key: value[i+1] for key, value in image_data.items()}
    model_manager._dirty = True
    blend_image, _ = model_manager.get_heatmap_vis(image1_point)
    heat_map_pred = model_manager._selected_model.get_heatmap(image1_point)
    heat_map_pred_r= np.resize(heat_map_pred, image1.shape[:2])
    blend_image_ = np.array(blend_image)[:,:,:1]
    pred_index = np.unravel_index(np.argmax(blend_image_, axis=None), blend_image_.shape)[:2]
    pred_index = (pred_index[1], pred_index[0])
    ground_truth_map = model_manager.create_ground_truth_map((r[0], r[1]))
    ground_truth_point = model_manager._transform_points((r[0], r[1]))
    plt.imshow(blend_image)
    plt.scatter(pred_index[0], pred_index[1], c='b', marker='x')
    plt.scatter(ground_truth_point[0],ground_truth_point[1], c='r', marker='x')
    plt.show()
    model_manager._dirty = False
    error = get_error(ground_truth_point, pred_index)
    print("error for corresponding image from reference image ", i, " is ", error)

