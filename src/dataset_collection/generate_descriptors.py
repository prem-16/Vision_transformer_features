import argparse 
import gzip
import pickle
from src.dataset_collection.generate_dataset import store_data
import numpy as np

from src.gui.main import model_manager
from src.models.model_gui_manager import ModelGUIManager


def generate_descriptors(model='DinoViT', stride=4, load_size=324, layer=11, facet='key', threshold=0.05, model_type='dino_vits8', log_bin=0, dataset_path='./test_data', descriptor_out='./test_data/descriptors'):
    """
    Generate the descriptors from the generated dataset.
    """
    model_manager = ModelGUIManager()
    model_manager.update_model(model)
    model_manager.apply_setting("stride", stride)
    model_manager.apply_setting("load_size", load_size)
    model_manager.apply_setting("layer", layer)
    model_manager.apply_setting("facet", facet)
    model_manager.apply_setting("threshold", threshold)
    model_manager.apply_setting("model_type", model_type)
    model_manager.apply_setting("log_bin", log_bin)
    descriptor_list = []
    # Load the dataset
    data_file = os.path.join(dataset_path, 'data.pkl.gzip')
    f = gzip.open(dataset_path, 'rb')
    data = pickle.load(f)
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
    store_data(descriptor_list, descriptor_out)


if __name__ == '__main__':
    # Use argparse to get the arguments
    arg = argparse.ArgumentParser()
    # Model 
    arg.add_argument('--model', type=str, default='DinoViT')
    # Stride
    arg.add_argument('--stride', type=int, default=4)
    # Load size
    arg.add_argument('--load_size', type=int, default=324)
    # Layer
    arg.add_argument('--layer', type=int, default=11)
    # Facet
    arg.add_argument('--facet', type=str, default='key')
    # Threshold
    arg.add_argument('--threshold', type=float, default=0.05)
    # Model Type
    arg.add_argument('--model_type', type=str, default='dino_vits8')
    # Log Bin
    arg.add_argument('--log_bin', type=int, default=0)
    # Data set path
    arg.add_argument('--dataset_path', type=str, default='./test_data')
    # Descriptor save output path
    arg.add_argument('--descriptor_out', type=str, default='./test_data/descriptors')
    # Parse the arguments
    args = arg.parse_args()

    # Generate the descriptors
    generate_descriptors(args.model, args.stride, args.load_size, args.layer, args.facet, args.threshold, args.model_type, args.log_bin, args.dataset_path, args.descriptor_out)
