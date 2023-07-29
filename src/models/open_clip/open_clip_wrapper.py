import numpy as np
import open_clip
import torch
from PIL import Image
from open_clip import image_transform
from torch.nn import Module

from src.models.model_wrapper import ModelWrapperBase


class DinoVITWrapper(ModelWrapperBase):
    NAME = "OPEN_CLIP"

    SETTINGS = {
        "model_type": {
            "type": "dropdown",
            "options": [
                ('convnext_base', 'laion400m_s13b_b51k'),
                ('convnext_base_w', 'laion2b_s13b_b82k'),
                ('convnext_base_w', 'laion2b_s13b_b82k_augreg'),
                ('convnext_base_w', 'laion_aesthetic_s13b_b82k'),
                ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'),
                ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'),
                ('convnext_large_d', 'laion2b_s26b_b102k_augreg'),
                ('convnext_large_d_320', 'laion2b_s29b_b131k_ft'),
                ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'),
                ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'),
                ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'),
                ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup')
            ],
            "default": ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg')
        },
        "layer": {
            "type": "hidden",
            "options": [
                "0",
                # 40x40 Bad
                "1",
                # 20x20 Good
                "2",
                # 10x10 Good/Ok
                "3"
            ],
            "default": "2"
        },
        "block": {
            "type": "hidden",
            "options": [
                "0", "1", "2"
            ],
            "default": "1"
        },
        "load_size": {
            "type": "hidden",
            "default": 320
        },
    }

    def __init__(self):
        super().__init__()
        self._cache = {}
        self._persistant_cache = {}

    def _get_model(self, **settings):
        with torch.no_grad():
            if self._persistant_cache.get('model', None) is None:
                # Load model
                model, _, _ = open_clip.create_model_and_transforms(settings["model_type"][0],
                                                                             pretrained=settings["model_type"][1])
                preprocess = image_transform(
                    settings['load_size'], is_train=False, mean=None, std=None
                )

                self._persistant_cache['model'] = model
                self._persistant_cache['preprocess'] = preprocess

                # Record gradients for the following layer:
                layer = model.visual.trunk.stages._modules[settings['layer']].blocks._modules[settings['block']]

                def hook_func(m, inp, op):
                    self._persistant_cache['feat'] = op.detach()

                layer.register_forward_hook(hook_func)
            else:
                model = self._persistant_cache['model']
                preprocess = self._persistant_cache['preprocess']
        return model, preprocess

    def _compute_descriptors(self, image_pil: Image.Image, **kwargs):
        """
        Computes the descriptors for the image.
        :param image_dir: The directory of the image.
        :param kwargs: A dictionary of settings for the model.
        :return: descriptors and dictionary of other information.
        """

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model
        model, preprocess = self._get_model(**kwargs)
        # Preprocess the image
        image = preprocess(image_pil).unsqueeze(0).to('cpu')

        # Encode the image using the image encoder portion
        model.encode_image(image)

        image_features = self._persistant_cache['feat']

        # image_feature is e.g. (1, desc_size, dim, dim)
        num_patches = (image_features.shape[-2], image_features.shape[-1])
        load_size = image.size

        # image_features = image_features.flatten(0, 1)
        # flatten the spatial dimensions to (1, 1, desc_size, num_patches)
        image_features = image_features.reshape(1, 1, image_features.shape[1], -1)
        # Permute the descriptors such that we have e.g. (1, 1, num_patches, desc_size)
        image_features = image_features.permute(0, 1, 3, 2)

        """
        
            num_patches = (descriptors.shape[-2], descriptors.shape[-1])

            # To prepare the descriptors we must reshape them such that we have all patches on the same axis
            # e.g. (1, 1280, 48, 48) -> (1, 1, 1280, 2304)
            descriptors = descriptors.reshape(1, 1, descriptors.shape[1], -1)
            # Permute the descriptors such that we have e.g. (1, 1, 2304, 1280)
            descriptors = descriptors.permute(0, 1, 3, 2)
        
        """
        # num_patches, load_size = extractor.num_patches, extractor.load_size

        return image_features, {
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
            # Extracting descriptors for each image
            model = self._get_model(**settings)

            # Compute the descriptors
            descriptor_dump_1 = self._compute_descriptors_from_dir(image_dir_1, **settings, model=model)
            descriptor_dump_2 = self._compute_descriptors_from_dir(image_dir_2, **settings, model=model)

            response = self._build_similarity_cache_from_descriptor_dump(descriptor_dump_1, descriptor_dump_2)

        return response

    def process_image_pair(self, image_dir_1, image_dir_2, settings=None):
        """
        Process the two images and return the similarity map.
        :param image_dir_1: The directory of the first image.
        :param image_dir_2: The directory of the second image.
        :param settings: A dictionary of settings for the model.
        :return: A dictionary of the descriptors, similarities and other information.
        """
        self._cache = self._get_descriptor_similarity(image_dir_1, image_dir_2, settings)
