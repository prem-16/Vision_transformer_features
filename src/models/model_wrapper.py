from abc import ABC

import numpy as np
import torch
from PIL import Image

from src.models.dino_vit.correspondences import chunk_cosine_sim


class ModelWrapperBase(ABC):
    """
    An interface to access the models and standardize their methods.
    """
    # The name of the model
    NAME = None
    # A dictionary of settings for the tkinter GUI
    SETTINGS = None

    @staticmethod
    def _compute_similarity(descriptors_1, descriptors_2):
        return chunk_cosine_sim(descriptors_1, descriptors_2)

    @classmethod
    def from_pkl(cls, desc_pkl_path_1, desc_pkl_path_2):
        """
        Build class and create cache from pkl.

        :param desc_pkl_path_1:
        :param desc_pkl_path_2:
        :return:
        """
        # Load descriptors from pickle files
        pkl_1 = torch.load(desc_pkl_path_1)
        pkl_2 = torch.load(desc_pkl_path_2)

        descriptors_1, other_info_1 = pkl_1
        descriptors_2, other_info_2 = pkl_2

        # Compute similarity
        similarity = cls._compute_similarity(descriptors_1, descriptors_2)

        # Build cache
        cache = {
            "descriptors_1": descriptors_1,
            "descriptors_2": descriptors_2,
            "similarity": similarity,
            "num_patches_1": other_info_1['num_patches'],
            "num_patches_2": other_info_2['num_patches']
        }

        instance = cls()
        instance._cache = cache
        return instance


    @classmethod
    def _compute_descriptors(cls, image: Image.Image, **kwargs):
        """
        Computes the descriptors for the image.
        :param image: The image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """
        raise NotImplementedError("Not implemented!")

    @classmethod
    def _compute_descriptors_from_numpy(cls, image: np.ndarray, **kwargs):
        """
        Computes the descriptors for the image.
        :param image: The image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """
        image = Image.fromarray(image)
        return cls._compute_descriptors(image, **kwargs)

    @classmethod
    def _compute_descriptors_from_dir(cls, image_dir, **kwargs):
        """
        Computes the descriptors for the image.
        :param image_dir: The directory of the image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """
        image = Image.open(image_dir).convert("RGB")
        return cls._compute_descriptors(image, **kwargs)

    @staticmethod
    def _get_descriptor_index_from_point(point, num_patches):
        """
        Converts a point in the image to a descriptor index.
        :param point: The point in the image.
        :param load_size: The size of the image.
        :param num_patches: The number of patches in the image.
        :return: The descriptor index.
        """
        point_x, point_y = point

        # Turn image pixel point to descriptor map point
        point_x = point_x * num_patches[1]
        point_y = point_y * num_patches[0]

        # Get the descriptor map point's index
        point_index = int(point_y) * num_patches[1] + int(point_x)
        return point_index

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
        Get heatmap / activation map for the second image.
        Seperate to get_heatmap_vis as this may be overrided.
        :param point: The point in the first image.
        """

        # Get the annotated descriptor index
        descriptor_index = self._get_descriptor_index_from_point(
            point, self._cache['num_patches_1']
        )

        # Filter similarity map to only show the similarity of the annotated descriptor
        similarity_map = self._cache['similarities'][0, 0, descriptor_index]

        # Softmax the similarity map
        similarity_map = torch.softmax(similarity_map, dim=0)

        # Normalize the similarity map
        similarity_map = (
            (similarity_map - torch.min(similarity_map)) / (torch.max(similarity_map) - torch.min(similarity_map))
        )

        # Convert the similarity map to a heatmap
        heatmap = similarity_map.view(self._cache['num_patches_2']).cpu().numpy()

        return heatmap

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
        return image_2 , heatmap

    def compute_descriptors_from_pkl_sequence(self, sequence_pkl_dir):
        """
        Compute descriptors from a sequence pkl file.
        :param sequence_pkl_dir: The directory of the sequence pkl file.
        :return: Descriptors
        """
        # TODO use _compute_descriptors
        pass

