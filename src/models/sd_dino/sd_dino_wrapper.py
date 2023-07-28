import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import PILToTensor

from sd_dino.extractor_sd import load_model, process_features_and_mask
from sd_dino.extractor_dino import ViTExtractor
from sd_dino.utils.utils_correspondence import co_pca, resize, co_pca_single_image
from src.models.dift.dift_sd import SDFeaturizer

from src.models.model_wrapper import ModelWrapperBase


class SDDINOWrapper(ModelWrapperBase):
    NAME = "SD_DINO"

    SETTINGS = {
        "combine_pca": {
            "type": "toggle",
            "default": False
        },
        "fuse_dino": {
            "type": "toggle",
            "default": True
        },
        "only_dino": {
            "type": "toggle",
            "default": False
        },
        "dino_v2": {
            "type": "toggle",
            "default": True
        },
        "t": {
            "type": "slider",
            "min": 1,
            "max": 1000,
            "default": 100,
            "step": 1
        },
        "load_size": {
            # Should be a multiple of 14 for DINOv2
            "type": "slider",
            "min": 1,
            "max": 1000,
            "default": 224
        },
        "sd_load_size": {
            "type": "hidden",
            "default": 960
        },
        "stride": {
            # Totally dependent on chosen settings
            "type": "hidden",
            "default": "empty"
        },
        "facet": {
            # Totally dependent on chosen settings
            "type": "hidden",
            "default": "empty"
        },
        "text_input": {
            "type": "hidden",
            "default": None
        },
        "pca_dims": {
            "type": "hidden",
            "default": [256, 256, 256]
        },
        "seed": {
            "type": "hidden",
            "default": 42
        },
        "edge_pad": {
            "type": "hidden",
            "default": False
        },
        "pca": {
            "type": "hidden",
            "default": True
        },
        "ver": {
            "type": "hidden",
            "default": "v1-5"
        },
        "model_type": {
            "type": "hidden",
            "default": "base"
        },
        "layer": {
            "type": "hidden",
            "default": "empty"
        }
    }

    def __init__(self):
        super().__init__()
        # This cache is used to store the descriptors and similarities for the current images
        # however it is not persistant between runs.
        self._cache = {}
        # This cache is used to keep the model in memory between runs.
        # As it takes minutes to load sd!
        self._persistant_cache = {}

    def _cache_models(self, **settings):
        with torch.no_grad():
            # If we have not run the cache operation yet...
            if self._persistant_cache.get('dino_extractor', None) is None:
                # ---- LOAD SD MODEL ----
                if settings['only_dino'] is False:
                    sd_model, sd_aug = load_model(
                        diffusion_ver=settings['ver'],
                        image_size=settings['sd_load_size'],
                        num_timesteps=settings['t']
                    )
                    self._persistant_cache['sd_model'] = sd_model
                    self._persistant_cache['sd_aug'] = sd_aug
                else:
                    self._persistant_cache['sd_model'] = None
                    self._persistant_cache['sd_aug'] = None

                # ---- LOAD DINO MODEL ----
                img_size = settings.get('load_size', 840 if settings['dino_v2'] else 244)
                model_dict = {
                    'small': 'dinov2_vits14',
                    'base': 'dinov2_vitb14',
                    'large': 'dinov2_vitl14',
                    'giant': 'dinov2_vitg14'
                }

                model_type = model_dict[settings['model_type']] if settings['dino_v2'] else 'dino_vits8'

                # Define layer
                layer = settings.get('layer', None)
                if layer in [None, 'empty']:
                    # Large
                    if 'l' in model_type:
                        layer = 23
                    # Giant
                    elif 'g' in model_type:
                        layer = 39
                    else:
                        layer = 11 if settings['dino_v2'] else 9
                if isinstance(layer, str):
                    layer = int(layer)

                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # Stride and facet are determined by the chosen settings
                # However they can be overrided.
                stride = settings.get('stride', None)
                facet = settings.get('facet', None)
                stride = stride if (stride not in [None, 'empty']) else (14 if settings['dino_v2'] else 4)
                if isinstance(stride, str):
                    stride = int(stride)

                facet = facet if (facet not in [None, 'empty']) else ('token' if settings['dino_v2'] else 'key')
                if isinstance(facet, str):
                    facet = int(facet)

                extractor = ViTExtractor(model_type, stride, device=device)

                patch_size = extractor.model.patch_embed.patch_size[
                    0] if settings['dino_v2'] else extractor.model.patch_embed.patch_size
                num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

                self._persistant_cache['dino_extractor'] = extractor
                self._persistant_cache['num_patches'] = num_patches
                self._persistant_cache['layer'] = layer
                self._persistant_cache['facet'] = facet
                self._persistant_cache['patch_size'] = patch_size
                self._persistant_cache['img_size'] = img_size
                self._persistant_cache['device'] = device

    def _compute_descriptors(self, image: Image.Image, **kwargs):

        is_dino_v2 = kwargs['dino_v2']
        is_dino_only = kwargs['only_dino']
        is_dino_fuse = kwargs['fuse_dino']
        is_combine_pca = kwargs['combine_pca']

        # Assert that if only_dino is True, then fuse_dino must be True
        if is_dino_only and not is_dino_fuse:
            raise ValueError("If only_dino is True, then fuse_dino must be True")

        dist = 'l2' if is_dino_fuse and not is_dino_only else 'cos'

        with torch.no_grad():
            self._cache_models(**kwargs)
            num_patches = self._persistant_cache['num_patches']
            layer = self._persistant_cache['layer']
            facet = self._persistant_cache['facet']
            dino_extractor = self._persistant_cache['dino_extractor']

            input_text = kwargs.get('text_input', None)

            # Resize the image
            input_image = image.convert('RGB')
            # Lossy resize input_image to load_size to ensure sd_model has no advantage...
            input_image = input_image.resize((kwargs['load_size'], kwargs['load_size']), Image.BILINEAR)
            # Get the SD input image
            sd_image = resize(input_image, kwargs['sd_load_size'], resize=True, to_pil=True, edge=kwargs['edge_pad'])
            # Get the DINO input image
            dino_image = resize(input_image, kwargs['load_size'], resize=True, to_pil=True, edge=kwargs['edge_pad'])

            with torch.no_grad():
                # Stable Diffusion
                if is_dino_only is False:
                    if is_combine_pca is False:
                        # Don't use PCA
                        # When PCA is True: shape (1, 384, 60, 60)
                        # When PCA is False: shape (1, 1024, 60, 60)
                        image_desc_sd = process_features_and_mask(
                            self._persistant_cache['sd_model'], self._persistant_cache['sd_aug'],
                            sd_image, input_text=input_text, mask=False, pca=kwargs['pca']
                        )
                    else:
                        # Use PCA
                        # Shape (1, 768, 60, 60)
                        image_desc_sd = process_features_and_mask(
                            self._persistant_cache['sd_model'], self._persistant_cache['sd_aug'],
                            sd_image, input_text=input_text, mask=False, raw=True
                        )
                        image_desc_sd = co_pca_single_image(image_desc_sd, kwargs['pca_dims'])

                    # Reshape (1, descriptor_size, 60, 60) to (1, descriptor_size, num_patches, num_patches)
                    image_desc_sd = torch.nn.functional.interpolate(
                        image_desc_sd, (num_patches, num_patches), mode='bilinear'
                    )

                    # Now perform remaining reshaping operations...
                    image_desc_sd = image_desc_sd.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)

                # DINO
                if is_dino_fuse:
                    image_batch = dino_extractor.preprocess_pil(dino_image)
                    image_desc_dino = dino_extractor.extract_descriptors(
                        image_batch.to(self._persistant_cache['device']), layer, facet
                    )

                if dist == 'l1' or dist == 'l2':                    # Normalize the features
                    image_desc_sd = image_desc_sd / image_desc_sd.norm(dim=-1, keepdim=True)
                    # If DINO
                    if is_dino_fuse:
                        image_desc_dino = image_desc_dino / image_desc_dino.norm(dim=-1, keepdim=True)

                # If SD + DINO
                if is_dino_fuse and not is_dino_only:
                    # Cat two features together
                    image_desc = torch.cat((image_desc_sd, image_desc_dino), dim=-1)
                elif is_dino_only:
                    # If DINO only
                    image_desc = image_desc_dino
                else:
                    # If SD only
                    image_desc = image_desc_sd

            return image_desc.cpu(), {
                "num_patches": dino_extractor.num_patches,
                "load_size": (dino_image.size[1], dino_image.size[0])
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
            # Compute the descriptors
            descriptor_dump_1 = self._compute_descriptors_from_dir(image_dir_1, **settings)
            descriptor_dump_2 = self._compute_descriptors_from_dir(image_dir_2, **settings)

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
