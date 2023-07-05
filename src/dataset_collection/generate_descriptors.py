import argparse
import gzip
import pickle
from src.dataset_collection.generate_dataset import store_data
import numpy as np

from src.gui.main import model_manager
from src.models.model_gui_manager import ModelGUIManager


def generate_descriptors(class_type=None, settings=None, dataset_path=None, descriptor_out=None):
    """
    Generate the descriptors from the generated dataset.
    """
    model_manager = ModelGUIManager()
    model_manager.update_model("DinoViT")
    model_manager.apply_setting("stride", 4)
    model_manager.apply_setting("load_size", 324)
    model_manager.apply_setting("layer", 11)
    model_manager.apply_setting("facet", "key")
    model_manager.apply_setting("threshold", 0.05)
    model_manager.apply_setting("model_type", "dino_vits8")
    model_manager.apply_setting("log_bin", 0)
    descriptor_list = []
    # Load the dataset
    data_file = gzip.open(dataset_path, 'rb')
    data = pickle.load(data_file)
    number_of_images = len(data)
    # Iterate over the images
    for i in range(number_of_images):
        #   Compute the descriptor
        img = np.array(data['image_rgb'][i])
        descriptor = model_manager._selected_model._compute_descriptors_from_numpy(
            img
        )
        #   Append descriptor to list
        descriptor_list.append(descriptor)
    # Save descriptors list
    store_data(descriptor_list, "./test_data/descriptors/")


if __name__ == '__main__':
    # Use argparse to get the arguments
    arg = argparse.ArgumentParser()
    # Class type
    arg.add_argument('--class_type', type=str)
    # Settings class
    arg.add_argument('--settings', type=str, default='settings')
    # Dataset path
    arg.add_argument('--dataset_path', type=str, default='dataset')
    # Descriptor output
    arg.add_argument('--descriptor_out', type=str, default='descriptors')
    # Parse the arguments
    args = arg.parse_args()

    # Generate the descriptors
    # TODO