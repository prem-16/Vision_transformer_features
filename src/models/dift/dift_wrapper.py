import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift.dift_sd import SDFeaturizer

from src.models.model_wrapper import ModelWrapperBase


class DIFTWrapper(ModelWrapperBase):
    NAME = "DIFT"

    SETTINGS = {
        "img_size": {
            "type": "slider",
            "min": 1,
            "max": 2000,
            "default": 768,
            "step": 1
        },
        "ensemble_size": {
            "type": "slider",
            "min": 1,
            "max": 100,
            "default": 1
        },
        "up_ft_index": {
            "type": "slider",
            "min": 0,
            "max": 3,
            "default": 1
        },
        "t": {
            "type": "slider",
            "min": 1,
            "max": 1000,
            "default": 261,
            "step": 1
        },
        "model_type": {
            "type": "dropdown",
            "options": [
                "stabilityai/stable-diffusion-2-1"
            ],
            "default": "stabilityai/stable-diffusion-2-1"
        },
        "prompt": {
            "type": "text",
            "default": ""
        },
    }

    def __init__(self):
        super().__init__()
        self._cache = {}

    @classmethod
    def _compute_descriptors(cls, image: Image.Image, **kwargs):
        with torch.no_grad():
            if kwargs.get('model', None) is None:
                dift = SDFeaturizer(kwargs['model_type'])
            else:
                dift = kwargs['model']
            if kwargs['img_size'] > 0:
                image = image.resize([kwargs['img_size'], kwargs['img_size']])
            img_tensor = (PILToTensor()(image) / 255.0 - 0.5) * 2
            descriptors = dift.forward(
                img_tensor,
                prompt=kwargs.get('prompt', ""),
                t=kwargs['t'],
                up_ft_index=kwargs['up_ft_index'],
                ensemble_size=kwargs['ensemble_size']
            )

            num_patches = (descriptors.shape[-2], descriptors.shape[-1])

            # To prepare the descriptors we must reshape them such that we have all patches on the same axis
            # e.g. (1, 1280, 48, 48) -> (1, 1, 1280, 2304)
            descriptors = descriptors.reshape(1, 1, descriptors.shape[1], -1)
            # Permute the descriptors such that we have e.g. (1, 1, 2304, 1280)
            descriptors = descriptors.permute(0, 1, 3, 2)

            return descriptors, {
                "num_patches": num_patches,
                "load_size": (image.size[1], image.size[0])
            }

    def _get_descriptor_similarity(self, image_dir_1, image_dir_2, settings=None):
        """
        Computed the cosine similarity between the descriptors of the two images.
        :param image_dir_1:
        :param image_dir_2:
        :param settings: A dictionary of settings for the model.
        :return: A dictionary of the descriptors, similarities and other information.
        """

        # Important if you don't want your GPU to blow up
        with torch.no_grad():
            dift = SDFeaturizer(settings['model_type'])

            # Compute the descriptors
            descriptors_1, other_info_1 = self._compute_descriptors_from_dir(image_dir_1, model=dift, **settings)
            descriptors_2, other_info_2 = self._compute_descriptors_from_dir(image_dir_2, model=dift, **settings)

            # Get the grid dim of patches
            num_patches_1 = other_info_1['num_patches']
            num_patches_2 = other_info_2['num_patches']

            # calculate similarity between image1 and image2 descriptors
            similarities = self._compute_similarity(descriptors_1, descriptors_2)

        return {
            "descriptors_1": descriptors_1,
            "descriptors_2": descriptors_2,
            "similarities": similarities,
            "num_patches_1": num_patches_1,
            "num_patches_2": num_patches_2,
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
