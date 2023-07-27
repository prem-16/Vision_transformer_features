import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import gzip
import argparse
from PIL import Image
performance_path = "result"



def get_image(dataset_path = "test_data/data_rotation_X_episode_10.pkl.gzip"):

    """Iterate through all files"""


    f = gzip.open(dataset_path, "rb")
    data = pickle.load(f)
    fig , ax = plt.subplots(50)
    for i in range(len(data["image_rgb"])):

        print("image",i)
        image_array = data["image_rgb"][i]
        image = Image.fromarray(image_array)
        image.save(f"Data_visuals/image_{i}.jpg")














if __name__ == "__main__":
    get_image()
