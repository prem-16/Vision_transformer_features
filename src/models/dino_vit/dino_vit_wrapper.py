import numpy as np
import torch

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
            "max": 12,
            "default": 4
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
            image1_batch, image1_pil = extractor.preprocess(image_dir_1, settings['load_size'])
            descriptors1 = extractor.extract_descriptors(image1_batch.to(device), settings['layer'], settings['facet'], True)
            num_patches1, load_size1 = extractor.num_patches, extractor.load_size
            image2_batch, image2_pil = extractor.preprocess(image_dir_2, settings['load_size'])
            descriptors2 = extractor.extract_descriptors(image2_batch.to(device), settings['layer'], settings['facet'], True)
            num_patches2, load_size2 = extractor.num_patches, extractor.load_size

            # extracting saliency maps for each image
            saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
            saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]
            # threshold saliency maps to get fg / bg masks
            fg_mask1 = saliency_map1 > settings['threshold']
            fg_mask2 = saliency_map2 > settings['threshold']

            # calculate similarity between image1 and image2 descriptors
            similarities = chunk_cosine_sim(descriptors1, descriptors2)

        return {
            "descriptors1": descriptors1,
            "descriptors2": descriptors2,
            "saliency_map1": saliency_map1,
            "saliency_map2": saliency_map2,
            "fg_mask1": fg_mask1,
            "fg_mask2": fg_mask2,
            "similarities": similarities,
            "num_patches1": num_patches1,
            "num_patches2": num_patches2,
            "load_size1": load_size1,
            "load_size2": load_size2,
            "image1_batch": image1_batch,
            "image2_batch": image2_batch,
            "image1_pil": image1_pil,
            "image2_pil": image2_pil,
            "extractor": extractor
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

    def _get_descriptor_index_from_point(self, point, load_size, num_patches):
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

    def get_heatmap(self, point):

        # Get the annotated descriptor index
        descriptor_index = self._get_descriptor_index_from_point(
            point, self._cache['load_size1'], self._cache['num_patches1']
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
        heatmap = similarity_map.view(self._cache['num_patches2']).cpu().numpy()

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
