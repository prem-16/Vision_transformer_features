import argparse
import gzip
import os
import pickle

import torch
from tqdm import tqdm

from src.dataset_collection.dataset_object_labels import dataset_object_labels
from src.dataset_collection.helpers import store_data
import numpy as np

from src.models.model_wrapper import ModelWrapperBase
from src.models.model_wrapper_list import MODEL_DICT

import time


def generate_descriptors(
        model_wrapper: ModelWrapperBase = None,
        dataset_path=None,
        descriptor_dir=None,
        identifier=None,
        disable_timestamp=False,
        ignore_duplicates=False,
        allow_text_captions=False,
        settings=None
):
    """
    Generate the descriptors from the generated dataset.
    """

    assert model_wrapper is not None, "Model wrapper is None."
    # Assert dataset path exists
    assert os.path.exists(dataset_path), "Dataset path does not exist."
    # Create descriptor dir
    if not os.path.exists(descriptor_dir):
        os.makedirs(descriptor_dir, exist_ok=True)

    # Get descriptor_filename
    dataset_dir, dataset_name = os.path.split(dataset_path)

    # If dataset_name is blank, i.e. {dataset_path}/ is provided.
    if dataset_name == '':
        # Get all the files in the directory
        dataset_filenames = os.listdir(dataset_dir)
        # Filter the files to only include .pkl.gzip files
        dataset_filenames = [f for f in dataset_filenames if f.endswith('.pkl.gzip')]
    else:
        dataset_filenames = [dataset_name]

    # For each dataset file, generate the descriptors...
    for dataset_name in tqdm(dataset_filenames):
        # Get the (new) dataset path
        dataset_path = os.path.join(dataset_dir, dataset_name)
        # Define descriptor filename with timestamp at end
        descriptor_filename = f"descriptor_{model_wrapper.NAME}"
        if identifier is not None:
            descriptor_filename = f"(id_{identifier})_{descriptor_filename}"
        if disable_timestamp is False:
            descriptor_filename = f"{descriptor_filename}_{time.strftime('%Y_%m_%d-%H_%M_%S')}"
        # Add the filename (with .pkl.gzip)
        descriptor_filename += f"_{dataset_name}"

        # If the descriptor already exists, and ignore_duplicates is True, then return
        if os.path.exists(os.path.join(descriptor_dir, descriptor_filename)) and ignore_duplicates is True:
            return

        # Get the episode number
        # If dataset_name contains "episode_", then get the following number
        if "episode_" in dataset_name:
            # Get the episode number
            episode_number = int(dataset_name.split("episode_")[1].split(".pkl.gzip")[0])
            # Get the object name from the episode
            object_name = dataset_object_labels[str(episode_number % len(dataset_object_labels))]
            # If allow_text_captions is True, then add the object name as a caption
            if allow_text_captions:
                settings['text_input'] = object_name

        # Load the dataset
        f = gzip.open(dataset_path, 'rb')
        data = pickle.load(f)
        number_of_images = len(data['image_rgb'])

        # Generate the descriptors
        descriptor_list = []
        for i in tqdm(range(number_of_images)):
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

        store_data(descriptor_save_dict, datasets_dir=descriptor_dir, descriptor_name=descriptor_filename)

    print("Max torch cuda memory allocated: %fGB" % (torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Use argparse to get the arguments
    arg = argparse.ArgumentParser()
    # Model 
    arg.add_argument('--model', type=str, required=True)
    # Data set path
    arg.add_argument('--dataset_path', type=str, required=True)
    # Descriptor save output path
    arg.add_argument('--descriptor_dir', type=str, required=True)
    # Optional identifier
    arg.add_argument('--identifier', type=str, default=None, required=False)
    # Disable timestamp
    arg.add_argument('--disable_timestamp', type=str2bool, default=False, required=False)
    # Ignore_duplicates
    arg.add_argument('--ignore_duplicates', type=str2bool, default=False, required=False)
    # Dataset text captions
    arg.add_argument('--allow_text_captions', type=str2bool, default=False, required=False)
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
            setting_type = str2bool
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
        identifier=args.pop('identifier', None),
        disable_timestamp=args.pop('disable_timestamp', False),
        ignore_duplicates=args.pop('ignore_duplicates', False),
        allow_text_captions=args.pop('allow_text_captions', False),
        settings=args
    )
