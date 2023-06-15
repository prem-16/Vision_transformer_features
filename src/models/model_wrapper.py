import numpy as np
from PIL import Image


class ModelWrapperBase:
    """
    An interface to access the models and standardize their methods.
    """
    # The name of the model
    NAME = None
    # A dictionary of settings for the tkinter GUI
    SETTINGS = None

    def process_image_pair(self, image_dir_1, image_dir_2, settings=None):
        """
        Compute a one time process of the two images.
        This is the feature extraction, similarity computation, etc.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :param settings: A dictionary of settings for the model.
        :return: None
        """
        raise NotImplementedError("Not implemented!")

    def get_heatmap(self, point):
        """
        Compute the heatmap of the second image using an annotated point from the first image.
        :return: A heatmap of the similarities of numpy array type.
        """
        raise NotImplementedError("Not implemented!")

    def get_heatmap_vis(self, image_dir_2, point):
        """
        Similar to get_heatmap but returns the heatmap overlayed on the second image.
        :return: The heatmap overlayed on the second image as a PIL image.
        """
        heatmap = self.get_heatmap(point)

        # Resize the heatmap to the size of the right image
        image_2 = Image.open(image_dir_2).convert("RGB")
        heatmap = heatmap.copy()

        # Heatmap is numpy array
        # First convert (H, W) to (H, W, 3)
        heatmap = np.expand_dims(heatmap, axis=2)
        # Then repeat the channels
        heatmap = np.repeat(heatmap, 3, axis=2)
        heatmap[:, :, 1:] = 0
        # Convert to PIL image
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        # Resize
        heatmap = heatmap.resize(image_2.size)
        # Add the heatmap values to the image and cap the values at 255
        image_2 = Image.blend(image_2, heatmap, 0.5)
        return image_2


class TestWrapper(ModelWrapperBase):
    NAME = "Test"

    SETTINGS = None
