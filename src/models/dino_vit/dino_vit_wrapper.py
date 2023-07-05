import numpy as np
import torch
from PIL import Image

from src.models.dino_vit.correspondences import chunk_cosine_sim
from src.models.dino_vit.extractor import ViTExtractor
from src.models.model_wrapper import ModelWrapperBase


class DinoVITWrapper(ModelWrapperBase):
    NAME = "DinoViT"

    SETTINGS = {
        "stride": {
            "type": "slider",
            "min": 1,
            "max": 10,
            "default": 4
        },
        "load_size": {
            "type": "slider",
            "min": 1,
            "max": 1000,
            "default": 224
        },
        "layer": {
            "type": "slider",
            "min": 1,
            "max": 11,
            "default": 9
        },
        "facet": {
            "type": "dropdown",
            "options": ["key", "query", "value", "token"],
            "default": "key"
        },
        "threshold": {
            "type": "slider",
            "min": 0,
            "max": 1.0,
            "default": 0.05,
            "step": 0.01
        },
        "log_bin": {
            "type": "slider",
            "min": 0,
            "max": 1,
            "default": 0
        },
        "model_type": {
            "type": "dropdown",
            "options": [
                "dino_vits8", "dino_vits16", "dino_vitb8", "dino_vitb16", "vit_small_patch8_224",
                "vit_small_patch16_224", "vit_base_patch8_224", "vit_base_patch16_224"
            ],
            "default": "dino_vits8"
        }
    }

    def __init__(self):
        super().__init__()
        self._cache = {}

    @classmethod
    def _compute_descriptors(cls, image_pil: Image.Image, **kwargs):
        """
        Computes the descriptors for the image.
        :param image_dir: The directory of the image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """

        with torch.no_grad():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(device)
            if kwargs.get('model', None) is None:
                extractor = ViTExtractor(kwargs['model_type'], kwargs['stride'], device=device)
            else:
                extractor = kwargs['model']
            image_batch, image_pil = extractor.preprocess_from_pil(image_pil, kwargs['load_size'])
            descriptors = extractor.extract_descriptors(
                image_batch.to(device), kwargs['layer'], kwargs['facet'], bin=kwargs['log_bin']
            )
            num_patches, load_size = extractor.num_patches, extractor.load_size

        return descriptors, {
            "num_patches": num_patches,
            "load_size": load_size
        }

    def _get_descriptor_similarity(self, image_dir_1, image_dir_2, settings=None):
        """
        Computed the cosine similarity between the descriptors of the two images.
        (Copied and pasted from correspondences.py)
        :param image_dir_1:
        :param image_dir_2:
        :param settings: A dictionary of settings for the model.
        :return: A dictionary of the descriptors, similarities and other information.
        """

        # Important if you don't want your GPU to blow up
        with torch.no_grad():
            # extracting descriptors for each image
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            extractor = ViTExtractor(settings['model_type'], settings['stride'], device=device)

            # Compute the descriptors
            descriptors_1, other_info_1 = self._compute_descriptors_from_dir(image_dir_1, model=extractor, **settings)
            descriptors_2, other_info_2 = self._compute_descriptors_from_dir(image_dir_2, model=extractor, **settings)

            # # extracting saliency maps for each image
            # saliency_map_1 = extractor.extract_saliency_maps(other_info_1['image_batch'].to(device))[0]
            # saliency_map_2 = extractor.extract_saliency_maps(other_info_2['image_batch'].to(device))[0]
            # # threshold saliency maps to get fg / bg masks
            # fg_mask_1 = saliency_map_1 > settings['threshold']
            # fg_mask_2 = saliency_map_2 > settings['threshold']

            # calculate similarity between image1 and image2 descriptors
            similarities = self._compute_similarity(descriptors_1, descriptors_2)

        return {
            "descriptors1": descriptors_1,
            "descriptors2": descriptors_2,
            "similarities": similarities,
            "num_patches_1": other_info_1['num_patches'],
            "num_patches_2": other_info_2['num_patches']
        }

    def process_image_pair(self, image_dir_1, image_dir_2, settings=None):
        """
        Process the two images and return the similarity map.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :param settings: A dictionary of settings for the model.
        :return: A dictionary of the descriptors, similarities and other information.
        """
        self._cache = self._get_descriptor_similarity(image_dir_1, image_dir_2, settings)

    def get_heatmap(self, point):

        # Get the annotated descriptor index
        descriptor_index = self._get_descriptor_index_from_point(
            point, self._cache['num_patches_1']
        )

        # Filter similarity map to only show the similarity of the annotated descriptor
        similarity_map = self._cache['similarities'][0, 0, descriptor_index]

        # TODO Maybe we need to look at all the similarities of image 2 to image 1?

        # Softmax the similarity map
        similarity_map = torch.softmax(similarity_map, dim=0)

        # Normalize the similarity map
        similarity_map = (
            (similarity_map - torch.min(similarity_map)) / (torch.max(similarity_map) - torch.min(similarity_map))
        )

        # Convert the similarity map to a heatmap
        heatmap = similarity_map.view(self._cache['num_patches_2']).cpu().numpy()

        return heatmap

# if __name__ == '__main__':
#     image = DinoVITWrapper().get_heatmap_vis(
#         "images/test_images/current_state.png",
#         "images/test_images/current_state.png",
#         (101, 59),
#         settings={
#             "model_type": "dino_vits8",
#             "stride": 4,
#             "layer": 9,
#             "facet": "key",
#             "threshold": 0.05,
#             "load_size": 224
#         }
#     )
#     # Convert image pil to numpy array
#     image = np.array(image)
#     pass
