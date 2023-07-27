import os
import pickle
import gzip
from PIL import Image
import shutil


def get_image_list(image_directory):
    """
    Recursively get all image files in the directory and subdirectories.
    :return:
    """
    image_files = []
    image_dirs = []
    shutil.rmtree(image_directory)
    os.mkdir(image_directory)
    image_data = read_data(datasets_dir="./test_data")
    for i in range(len(image_data['image_rgb'])):
        im = Image.fromarray(image_data['image_rgb'][i])
        im.save(os.path.join(image_directory, f"test_{(i+1):02}.png"))
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(file)
                image_dirs.append(os.path.join(root, file))

    return sorted(image_files), sorted(image_dirs)


def read_data(datasets_dir="./test_data", dataset_name="data.pkl.gzip"):
    """
    This method reads the states from the data file(data.pkl.gzip).
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, dataset_name)

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)
    print(len(data['image_rgb']))
    # get images as features and actions as targets

    sample = {
        "image_rgb": data['image_rgb'],
        "extrinsic": data['extrinsic'],
        "intrinsic": data['intrinsic'],
        "depth": data['depth'],

    }
    return data

def store_data(data, datasets_dir="./test_data", file_name='data_rotation_apple.pkl.gzip'):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, file_name)
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()