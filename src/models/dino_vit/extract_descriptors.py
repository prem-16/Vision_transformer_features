from extractor import ViTExtractor
import argparse
import torchvision.transforms as transforms
from PIL import Image
import torch
import tkinter as tk

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate ViT Descriptor extraction.')
    parser.add_argument('--image_path', type=str, required=True, help='path of the extracted image.')
    parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                              small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")

    args = parser.parse_args()

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        extractor = ViTExtractor(args.model_type, args.stride, device=device)
        image_batch, image_pil = extractor.preprocess(args.image_path, args.load_size)
        B, C, H, W = image_batch.shape
        HP, WP = H // extractor.p, W // extractor.p
        print(f"Image {args.image_path} is preprocessed to tensor of size {image_batch.shape}.")
        descriptors = extractor.extract_descriptors(image_batch.to(device), args.layer, args.facet, args.bin)
        print(f"Descriptors are of size: {descriptors.shape}")
        torch.save(descriptors, args.output_path)
        output = descriptors.view(B, 1, HP, WP, 384)
        print(f"Descriptors after reshaping: {output.shape}")
        print(f"Descriptors saved to: {args.output_path}")