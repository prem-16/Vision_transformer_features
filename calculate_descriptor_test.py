import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import gzip
import torch
from src.models.dino_vit.dino_vit_wrapper import DinoVITWrapper



if __name__ == "__main__":
     
    print('read data')
    data_file = os.path.join('./test_data_2', 'data.pkl.gzip')
    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)
    number_of_imgs = len(data['image_rgb'])
    #number_of_imgs = 5
    for i in np.arange(0, number_of_imgs):
        img = np.array(data['image_rgb'][i])
        plt.imshow(img)
        plt.show()
        
        #descriptors_list = np.array([])
        descriptor, meta = DinoVITWrapper()._compute_descriptors_from_numpy(
                img,
                model_type = "dino_vits8",
                stride = 4,
                layer = 9,
                facet = "key",
                threshold = 0.05,
                load_size = 224,
                log_bin = 0
        )
        print(f'num_patches: {meta["num_patches"]}')
        print(f'load_size: {meta["load_size"]}')
        print(f"Descriptors are of size: {descriptor.shape}")
        #descriptors_list = np.append(descriptors_list, descriptor.cpu())
        save_path = "./test_data_2/descriptor_{}.pth".format(i)
        torch.save(descriptor, save_path)
        print(f"Descriptor_{i} saved")

        #t = torch.load(save_path)
        #print(f"check size: {t.shape}")
    #np.save("./test_data_2/descriptors.npy", descriptors_list) 
    
    #desc = np.load("./test_data_2/descriptors.npy")
    #print(desc.size)
