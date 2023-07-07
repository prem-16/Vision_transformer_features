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
            "default": True
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
        }
    }

    OTHER_SETTINGS = {
        "ver": "v1-5",
        "pca_dims": [256, 256, 256],
        "size": 960,
        "model_size": "base",
        "text_input": None,
        "seed": 42,
        "edge_pad": False,
        "pca": False
    }

    def __init__(self):
        super().__init__()
        self._cache = {}

    @classmethod
    def _get_sd_model(cls, **settings):
        np.random.seed(settings['seed'])
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed(settings['seed'])
        torch.backends.cudnn.benchmark = True

        if settings['only_dino'] is True:
            print("Skipping SD model...")
            return None, None
        else:
            print("Loading SD model...")
            with torch.no_grad():
                model, aug = load_model(
                    diffusion_ver=settings['ver'],
                    image_size=settings['size'],
                    num_timesteps=settings['t']
                )
            return model, aug

    @classmethod
    def _compute_descriptors(cls, image: Image.Image, **kwargs):

        # Combine **kwargs with OTHER_SETTINGS, ensuring that kwargs takes precedence
        kwargs = {**cls.OTHER_SETTINGS, **kwargs}

        is_dino_v2 = kwargs['dino_v2']
        is_dino_only = kwargs['only_dino']
        is_dino_fuse = kwargs['fuse_dino']
        is_combine_pca = kwargs['combine_pca']

        # Assert that if only_dino is True, then fuse_dino must be True
        if is_dino_only and not is_dino_fuse:
            raise ValueError("If only_dino is True, then fuse_dino must be True")

        dist = 'l2' if is_dino_fuse and not is_dino_only else 'cos'

        with torch.no_grad():
            if kwargs.get('model', None) is None:
                sd_model, aug = cls._get_sd_model(**kwargs)
            else:
                sd_model, aug = kwargs['model']

            # Compute the descriptors
            img_size = 840 if kwargs['dino_v2'] else 244
            model_dict = {
                'small': 'dinov2_vits14',
                'base': 'dinov2_vitb14',
                'large': 'dinov2_vitl14',
                'giant': 'dinov2_vitg14'
            }

            model_type = model_dict[kwargs['model_size']] if is_dino_v2 else 'dino_vits8'
            layer = 11 if is_dino_v2 else 9

            if 'l' in model_type:
                layer = 23
            elif 'g' in model_type:
                layer = 39
            facet = 'token' if is_dino_v2 else 'key'
            stride = 14 if is_dino_v2 else 4
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            extractor = ViTExtractor(model_type, stride, device=device)
            patch_size = extractor.model.patch_embed.patch_size[
                0] if is_dino_v2 else extractor.model.patch_embed.patch_size
            num_patches = int(patch_size / stride * (img_size // patch_size - 1) + 1)

            input_text = kwargs['text_input']

            # Resize the image
            image = image.convert('RGB')
            real_size = image.size[0]
            image_input = resize(image, real_size, resize=True, to_pil=True, edge=kwargs['edge_pad'])
            image = resize(image, img_size, resize=True, to_pil=True, edge=kwargs['edge_pad'])

            with torch.no_grad():
                if not is_combine_pca:
                    if not is_dino_only:
                        image_desc = process_features_and_mask(
                            sd_model, aug, image_input, input_text=input_text, mask=False, pca=kwargs['pca']
                        ).reshape(1, 1, -1, num_patches ** 2).permute(
                            0, 1, 3, 2
                        )
                    if is_dino_fuse:
                        image_batch = extractor.preprocess_pil(image)
                        image_desc_dino = extractor.extract_descriptors(image_batch.to(device), layer, facet)

                else:
                    if not is_dino_only:
                        features = process_features_and_mask(
                            sd_model, aug, image_input, input_text=input_text, mask=False, raw=True
                        )
                        processed_features1 = co_pca_single_image(features, kwargs['pca_dims'])
                        image_desc = processed_features1.reshape(1, 1, -1, num_patches ** 2).permute(0, 1, 3, 2)
                    if is_dino_fuse:
                        image_batch = extractor.preprocess_pil(image)
                        image_desc_dino = extractor.extract_descriptors(image_batch.to(device), layer, facet)

                if dist == 'l1' or dist == 'l2':
                    # Normalize the features
                    image_desc = image_desc / image_desc.norm(dim=-1, keepdim=True)
                    if is_dino_fuse:
                        image_desc_dino = image_desc_dino / image_desc_dino.norm(dim=-1, keepdim=True)

                if is_dino_fuse and not is_dino_only:
                    # Cat two features together
                    image_desc = torch.cat((image_desc, image_desc_dino), dim=-1)

                if is_dino_only:
                    image_desc = image_desc_dino

            return image_desc.cpu(), {
                "num_patches": extractor.num_patches,
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

        # Combine **kwargs with OTHER_SETTINGS, ensuring that settings takes precedence
        kwargs = {**self.OTHER_SETTINGS, **settings}

        # Important if you don't want your GPU to blow up
        with torch.no_grad():
            sd_model_contents = self._get_sd_model(**kwargs)

            # Compute the descriptors
            descriptor_dump_1 = self._compute_descriptors_from_dir(image_dir_1, model=sd_model_contents, **settings)
            descriptor_dump_2 = self._compute_descriptors_from_dir(image_dir_2, model=sd_model_contents, **settings)

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
