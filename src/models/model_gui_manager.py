import gzip
import pickle

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
        image2_points[1] = min(int(image2_points[1]), gt_map.shape[0] -1)
        image2_points[0] = min(int(image2_points[0]), gt_map.shape[1] -1)
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
        cache = self._build_similarity_cache_from_descriptor_dump(descriptors[0], descriptors[ref_index])

        # Set cache
        self.selected_model._cache = cache
        self._settings = settings

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

    @staticmethod
    def _compute_similarity(descriptors_1, descriptors_2):
        return chunk_cosine_sim(descriptors_1, descriptors_2)


