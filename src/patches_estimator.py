import torch
from sd_dino.extractor_dino import ViTExtractor

# # STRIDE = 14 if DINO_V2 else 4

# Patch size: 14
LOAD_SIZE = 448
STRIDE = 1
DINO_V2 = True

# # Patch size: 8
# LOAD_SIZE = 512
# STRIDE = 4
# DINO_V2 = False


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_dict = {
    'small': 'dinov2_vits14',
    'base': 'dinov2_vitb14',
    'large': 'dinov2_vitl14',
    'giant': 'dinov2_vitg14'
}

if DINO_V2:
    model_type = model_dict['base']
else:
    model_type = 'dino_vits8'

# Large
if 'l' in model_type:
    layer = 23
# Giant
elif 'g' in model_type:
    layer = 39
else:
    layer = 11 if DINO_V2 else 9

extractor = ViTExtractor(model_type, STRIDE, device=device)

patch_size = extractor.model.patch_embed.patch_size[0] if DINO_V2 else extractor.model.patch_embed.patch_size

num_patches = int(patch_size / STRIDE * (LOAD_SIZE // patch_size - 1) + 1)

print("Patch size: ", patch_size)
print("Num patches: ", num_patches)
