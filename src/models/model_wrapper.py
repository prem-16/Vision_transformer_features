from abc import ABC

import numpy as np
import torch
from PIL import Image
import gzip
import pickle
import os
from src.models.dino_vit.correspondences import chunk_similarity


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
        return chunk_similarity(descriptors_1, descriptors_2)

    @classmethod
    def _build_similarity_cache_from_descriptor_dump(cls, descriptor_dump_1, descriptor_dump_2):
        """
        Build cache from the compute descriptor dump, this includes computing the similarities.
        :param descriptor_dump_1:
        :param descriptor_dump_2:
        :return:
        """

        descriptors_1, other_info_1 = descriptor_dump_1
        descriptors_2, other_info_2 = descriptor_dump_2

        # Get the grid dim of patches
        num_patches_1 = other_info_1['num_patches']
        num_patches_2 = other_info_2['num_patches']

        # calculate similarity between image1 and image2 descriptors
        similarities = cls._compute_similarity(descriptors_1, descriptors_2)

        return {
            "descriptors_1": descriptors_1,
            "descriptors_2": descriptors_2,
            "similarities": similarities,
            "num_patches_1": num_patches_1,
            "num_patches_2": num_patches_2,
        }

    def build_cache_from_pkl_gzip(self, pkl_path, ref_index):
        """
        Build class and create cache from pkl gzip file.

        :param desc_pkl_path_1:
        :param desc_pkl_path_2:
        :return:
        """

        # Load descriptors from pickle files
        #   Extract gzip
        f = gzip.open(pkl_path, 'rb')
        #   Load pkl
        pkl = pickle.load(f)

        # Build cache
        cache = self._build_similarity_cache_from_descriptor_dump(pkl[0], pkl[ref_index])

        # Set cache
        self._cache = cache

    def _compute_descriptors(self, image: Image.Image, **kwargs):
        """
        Computes the descriptors for the image.
        :param image: The image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """
        raise NotImplementedError("Not implemented!")

    def _compute_descriptors_from_numpy(self, image: np.ndarray, **kwargs):
        """
        Computes the descriptors for the image.
        :param image: The image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """
        image = Image.fromarray(image)
        return self._compute_descriptors(image, **kwargs)

    def _compute_descriptors_from_dir(self, image_dir, **kwargs):
        """
        Computes the descriptors for the image.
        :param image_dir: The directory of the image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """
        image = Image.open(image_dir).convert("RGB")
        return self._compute_descriptors(image, **kwargs)

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

    def get_heatmap(self, point, temperature=1.0):
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
        similarity_map = torch.softmax(similarity_map / temperature, dim=0)

        # Normalize the similarity map
        similarity_map = (
            (similarity_map - torch.min(similarity_map)) / (torch.max(similarity_map) - torch.min(similarity_map))
        )
        similarity_map = similarity_map / torch.sum(similarity_map)

        # Convert the similarity map to a heatmap
        heatmap = similarity_map.view(self._cache['num_patches_2']).cpu().numpy()

        return heatmap

    def get_heatmap_vis_from_pil(self, image_2: Image.Image, point, heatmap=None):
        if heatmap is None:
            heatmap = self.get_heatmap(point, temperature=0.1)

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
        return image_2, heatmap

    def get_heatmap_vis_from_numpy(self, image_2: np.ndarray, point):
        return self.get_heatmap_vis_from_pil(Image.fromarray(image_2), point)

    def get_heatmap_vis(self, image_dir_2, point):
        """
        Similar to get_heatmap but returns the heatmap overlayed on the second image.
        :return: The heatmap overlayed on the second image as a PIL image.
        """
        heatmap = self.get_heatmap(point)

        # Resize the heatmap to the size of the right image
        image_2 = Image.open(image_dir_2).convert("RGB")
        heatmap = heatmap.copy()

        return self.get_heatmap_vis_from_pil(image_2, point, heatmap=heatmap)
