import gzip
import pickle

import torch

from src.models.model_wrapper_list import MODEL_DICT
import numpy as np
from src.models.dino_vit.correspondences import chunk_cosine_sim


class ModelGUIManager:
    """
    A class to manage the models, their settings and further
    provide an interface for the GUI to interact with the models.
    """

    def __init__(self):
        self._settings = None
        self.selected_model = None
        # A flag to indicate that processing is required.
        self._dirty = True
        self._image_dir_1 = None
        self._image_dir_2 = None
        self._image_data_1 = None
        self._image_data_2 = None
        self.super_cache = None

    @property
    def model_name(self):
        return self.selected_model.NAME

    @property
    def model(self):
        return self.selected_model

    @property
    def image_dir_1(self):
        return self._image_dir_1

    @image_dir_1.setter
    def image_dir_1(self, image_dir_1):
        if self._image_dir_1 != image_dir_1:
            self._image_dir_1 = image_dir_1
            self._set_dirty()

    @property
    def image_dir_2(self):
        return self._image_dir_2

    @image_dir_2.setter
    def image_dir_2(self, image_dir_2):
        if self._image_dir_2 != image_dir_2:
            self._image_dir_2 = image_dir_2
            self._set_dirty()

    def _set_dirty(self):
        self._dirty = True

    def _transform_points(self, image1_points):
        """
        Transform the points from image 1 to image 2.
        :param image1_data: The data of image 1.
        :param image2_data: The data of image 2.
        :param image1_points: The points in image 1.
        :return: The points in image 2.
        """

        # # Transform the points
        depth = float(self._image_data_1['depth'][image1_points[1]][image1_points[0]])
        image1_points_h = np.array([image1_points[0] * depth, image1_points[1] * depth, depth])
        image1_points_camera = np.matmul(np.linalg.inv(self._image_data_1['intrinsic']), image1_points_h)
        image1_points_camera_h = np.append(image1_points_camera, 1)
        image1_points_world_h = np.matmul(np.linalg.inv(self._image_data_1['extrinsic']), image1_points_camera_h)

        image2_points_camera_h = np.matmul(self._image_data_2['extrinsic'], image1_points_world_h)
        image2_points_camera = image2_points_camera_h[:3] / image2_points_camera_h[3]
        image2_points_h = np.matmul(self._image_data_2['intrinsic'], image2_points_camera)
        image2_points = image2_points_h[:2] / image2_points_h[2]

        return image2_points

    def create_ground_truth_map(self, image1_points):
        image2_points = self._transform_points(image1_points)
        gt_map = np.zeros(self._image_data_2['image_rgb'].shape[:2])
        image2_points[1] = max(min(int(image2_points[1]), gt_map.shape[0] - 1), 0)
        image2_points[0] = max(min(int(image2_points[0]), gt_map.shape[1] - 1), 0)
        gt_map[int(image2_points[1])][int(image2_points[0])] = 1
        return gt_map

    def update_model(self, model_name):
        assert model_name in MODEL_DICT, f"Model {model_name} not found!"
        self.selected_model = MODEL_DICT[model_name]()
        self._settings = {}
        self._set_dirty()

    def apply_setting(self, setting_name, setting_value):
        if self._settings.get(setting_name, None) != setting_value or setting_name not in self._settings:
            self._settings[setting_name] = setting_value
            print("Applied setting ", setting_name, " with value ", setting_value)
            self._set_dirty()

    def process_images(self):
        """
        Compute a one time process of the two images.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :return: None
        """
        self.selected_model.process_image_pair(self._image_dir_1, self._image_dir_2, self._settings)
        self._dirty = False

    def get_heatmap_vis(self, point):
        """
        Compute the heatmap of the second image.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :param point: The point to compare to.
        :return: The heatmap visualization of the second image.
        """
        if self._dirty:
            self.process_images()
        vis, heatmap = self.selected_model.get_heatmap_vis(self._image_dir_2, point)
        return vis, heatmap

    def resize_descriptor(self, descriptor, target_num_patches):
        # Descriptor will be of size (1, 1, num_patches**2, descriptor_size)
        # Will return size (1, 1, target_num_patches**2, descriptor_size)
        descriptor_orig_shape = list(descriptor.shape)
        original_num_patches = int(np.sqrt(descriptor_orig_shape[2]))
        # Original size but in spatial shape
        descriptor_original_spatial_shape = (
            descriptor_orig_shape[0],
            descriptor_orig_shape[1],
            original_num_patches,
            original_num_patches,
            descriptor_orig_shape[-1]
        )
        # Target size but in spatial shape
        descriptor_target_spatial_shape = (
            descriptor_orig_shape[0],
            descriptor_orig_shape[1],
            target_num_patches * target_num_patches,
            descriptor_orig_shape[-1]
        )
        return torch.nn.functional.interpolate(
            # Reshape additional descriptor to spatial shape (1, 1, dim, dim, descriptor_size)
            descriptor.reshape(descriptor_original_spatial_shape)[0].permute(0, 3, 1, 2),
            (target_num_patches, target_num_patches), mode='bilinear'
        ).permute(0, 2, 3, 1).reshape(descriptor_target_spatial_shape)

    def build_super_cache(self, pkl_paths, target_num_patches=120):
        if isinstance(pkl_paths, str):
            pkl_paths = [pkl_paths]
        assert isinstance(pkl_paths, list), "pkl_path must be a list of paths."
        cache_contents = None
        for pkl_path in pkl_paths:
            # Load descriptors from pickle files
            #   Extract gzip
            f = gzip.open(pkl_path, 'rb')
            #   Load pkl
            pkl = pickle.load(f)
            assert pkl is not None, "pkl file not loaded"
            print(len(pkl['descriptors'][0][0]))

            # Resize descriptors here
            pkl['descriptors'] = [
                (self.resize_descriptor(descriptor, target_num_patches), meta)
                for (descriptor, meta) in pkl['descriptors']
            ]

            if cache_contents is None:
                cache_contents = pkl
            else:
                descriptor_shape_config_1 = list(cache_contents['descriptors'][0][0].shape)
                descriptor_shape_config_2 = list(pkl['descriptors'][0][0].shape)

                print("descriptor_orig_shape_config_1", descriptor_shape_config_1)
                print("descriptor_orig_shape_config_2", descriptor_shape_config_2)

                # For each descriptor...
                for i in range(len(pkl['descriptors'])):
                    # Concatenate the descriptors on the descriptor axis
                    # And reshape to combined_new_shape
                    temp = list(cache_contents['descriptors'][i])

                    temp[0] = torch.concatenate(
                        (
                            # Reshape base descriptor to spatial shape (1, 1, dim, dim, descriptor_size)
                            cache_contents['descriptors'][i][0],
                            # Resize additional descriptor to spatial shape of base descriptor
                            pkl['descriptors'][i][0]
                        ),
                        axis=-1
                    )
                    cache_contents['descriptors'][i] = tuple(temp)

        for descriptors in cache_contents['descriptors']:
            descriptors[1]['num_patches'] = (target_num_patches, target_num_patches)
        self.super_cache = cache_contents

    def build_cache_from_pkl_gzip(self, pkl_paths, ref_index, point=None):
        """
        Build class and create cache from pkl gzip file.

        :param desc_pkl_path_1:
        :return:
        """
        if self.super_cache is None:
            self.build_super_cache(pkl_paths)

        #   Load pkl
        pkl = self.super_cache

        # Get the contents of the pkl
        if isinstance(pkl, dict):
            descriptors = pkl['descriptors']
            settings = pkl['settings']
        elif isinstance(pkl, list):
            descriptors = pkl
            settings = None
        else:
            raise ValueError("Incorrect type.")
        # Build cache
        print(point)
        cache = self._build_similarity_cache_from_descriptor_dump(descriptors[0], descriptors[ref_index], point=point)

        # Set cache
        self.selected_model._cache = cache
        self._settings = settings

    @classmethod
    def _build_similarity_cache_from_descriptor_dump(cls, descriptor_dump_1, descriptor_dump_2, point=None):
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
        print(
            point
        )
        # calculate similarity between image1 and image2 descriptors
        if point is None:
            index = None
        else:
            index = cls._get_descriptor_index_from_point(
                point, num_patches_1
            )
        similarities = cls._compute_similarity(descriptors_1, descriptors_2, index=index)

        return {
            "descriptors_1": descriptors_1,
            "descriptors_2": descriptors_2,
            "similarities": similarities,
            "num_patches_1": num_patches_1,
            "num_patches_2": num_patches_2,
        }

    @staticmethod
    def _compute_similarity(descriptors_1, descriptors_2, index=None):
        print(index)
        if index is None:
            return chunk_cosine_sim(descriptors_1, descriptors_2)
        else:

            return chunk_cosine_sim(descriptors_1, descriptors_2, index)

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
