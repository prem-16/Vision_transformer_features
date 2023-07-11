import argparse
import gzip
import os
import pickle

import torch
from tqdm import tqdm

from src.dataset_collection.helpers import store_data
import numpy as np

from src.models.model_wrapper import ModelWrapperBase
from src.models.model_wrapper_list import MODEL_DICT

import time

def generate_descriptors(
        model_wrapper: ModelWrapperBase = None,
        dataset_path=None,
        descriptor_dir=None,
        settings=None
):
    """
    Generate the descriptors from the generated dataset.
    """

    assert model_wrapper is not None, "Model wrapper is None."
    # Print current path
    print("Current path: %s"%(os.getcwd()))
    # Assert dataset path exists
    assert os.path.exists(dataset_path), "Dataset path does not exist."
    # Assert descriptor path exists
    assert os.path.exists(descriptor_path), "Descriptor path does not exist."

    descriptor_list = []
    # Load the dataset
    f = gzip.open(dataset_path, 'rb')
    data = pickle.load(f)
    number_of_images = len(data)
    # Iterate over the images
    for i in tqdm(range(number_of_images)):
        # Every n images, print the memory usage
        if i % 1 == 0:
            print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.max_memory_allocated(0)/1024/1024/1024))
        #   Compute the descriptor
        img = np.array(data['image_rgb'][i])
        descriptor = model_wrapper._compute_descriptors_from_numpy(img, **settings)
        #   Append descriptor to list
        descriptor_list.append(descriptor)

    # Create the descriptor save dictionary
    descriptor_save_dict = {
        "descriptors": descriptor_list,
        "settings": settings
    }

    # Split the descriptor path into path and file name
    descriptor_dir, descriptor_filename = os.path.split(descriptor_path)
    _, dataset_name = os.path.split(dataset_path)
    # Define descriptor filename with timestamp at end
    descriptor_filename = dataset_name.replace("data_", f"descriptor_{time.strftime('%Y_%m_%d-%H_%M_%S')}")
    store_data(descriptor_save_dict, descriptor_dir, descriptor_name=descriptor_filename)


if __name__ == '__main__':
    # Use argparse to get the arguments
    arg = argparse.ArgumentParser()
    # Model 
    arg.add_argument('--model', type=str, default='DinoViT')
    # Data set path
    arg.add_argument('--dataset_path', type=str, default='./test_data/data.pkl.gzip')
    # Descriptor save output path
    arg.add_argument('--descriptor_dir', type=str, default='./test_data/descriptors')
    known_args = arg.parse_known_args()[0]

    # Get the model wrapper
    model_wrapper = MODEL_DICT[known_args.model]()

    # Get the model wrapper SETTINGS
    model_wrapper_settings = model_wrapper.SETTINGS
    # For each setting, add it to the arg parser
    for setting_name, setting_content in model_wrapper_settings.items():
        # Derive the arg type
        tkinter_type = setting_content.get('type', None)
        if tkinter_type == 'slider':
            setting_type = int
        elif tkinter_type == 'dropdown':
            setting_type = str
        elif tkinter_type == 'toggle':
            setting_type = bool
        elif tkinter_type == 'text':
            setting_type = str
        elif tkinter_type == 'hidden':
            # Setting type is type of default value
            setting_type = type(setting_content.get('default', None))

        # Add the argument
        arg.add_argument(
            '--{}'.format(setting_name),
            type=setting_type,
            default=setting_content.get('default', None)
        )

    # Parse the arguments
    args = vars(arg.parse_args())
    # Remove this as model name is used later on in the model wrapper.
    args.pop('model', None)

    print("Settings:")
    for arg_name, arg_value in args.items():
        print(f"    {arg_name}: {arg_value}")

    # Generate the descriptors
    generate_descriptors(
        model_wrapper=model_wrapper,
        dataset_path=args.pop('dataset_path', None),
        descriptor_dir=args.pop('descriptor_dir', None),
        settings=args
    )
