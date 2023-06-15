from extractor import ViTExtractor
import torchvision.transforms as transforms
from PIL import Image
import torch

img = Image.open("images/cat.jpg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = ViTExtractor(device=device)
transform = transforms.Compose([transforms.ToTensor()])
imgs = torch.unsqueeze(transform(img).to(device),dim=0)

# imgs should be imagenet normalized tensors. shape BxCxHxW
descriptors = extractor.extract_descriptors(imgs)