from PIL import Image


class ModelWrapperBase:
    """
    An interface to access the models and standardize their methods.
    """
    # The name of the model
    NAME = None
    # A dictionary of settings for the tkinter GUI
    SETTINGS = None

    def get_correspondences_from_pair_array(self, image_1, image_2):
        """
        Get the correspondences between the two images.
        """
        raise NotImplementedError("Not implemented!")

    def get_correspondences_from_pair_dir(self, image_1_dir, image_2_dir):
        """
        Get the correspondences between the two images from the image directories.
        """
        # Get the images from the directories
        image_1 = Image.open(image_1_dir)
        image_2 = Image.open(image_2_dir)

        # Get the correspondences
        return self.get_correspondences_from_pair_array(image_1, image_2)

    def get_heatmap_from_pair_array(self, image_1, image_2):
        """
        ...
        """
        raise NotImplementedError("Not implemented!")
