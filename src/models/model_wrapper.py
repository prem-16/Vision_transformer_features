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
        image_2 = Image.open(image_dir_2)
        heatmap = heatmap.copy()
        # Convert heatmap to PIL image
        heatmap = Image.fromarray(heatmap)
        # Resize the PIL heatmap
        heatmap = heatmap.resize(image_2.size)

        # Overlay the heatmap on the right image
        # Convert the heatmap to RGB with only the red channel
        heatmap = heatmap.convert("RGB")
        heatmap = heatmap.split()[0]
        heatmap.putalpha(128)
        image_2.paste(heatmap, (0, 0), heatmap)

        return image_2


class TestWrapper(ModelWrapperBase):
    NAME = "Test"

    SETTINGS = None
