from PIL import Image
from torchvision.transforms import PILToTensor
from models.dift.dift_sd import SDFeaturizer

model_id = "stabilityai/stable-diffusion-2-1"
input_path = "images/test_images/were_rabbit.jpg"
img_size = [768, 768]
prompt = ""
t = 261
up_ft_index = 1
ensemble_size = 1
output_path = "dift.pt"

dift = SDFeaturizer(model_id)
img = Image.open(input_path).convert('RGB')
if img_size[0] > 0:
    img = img.resize(img_size)
img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
ft = dift.forward(img_tensor,
                  prompt=prompt,
                  t=t,
                  up_ft_index=up_ft_index,
                  ensemble_size=ensemble_size)
print(ft.shape)
