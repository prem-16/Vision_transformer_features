# Important! The test data dir must have a / at the end if you wish for it to handle multi-files...
pip uninstall open-clip-torch -y
pip install -r requirements_model_sd_dino.txt
sh ./generate_descriptors.sh ../../heisenberg/Vision_transformer_features/test_data/ ./descriptors ./experiments_sd_dino.txt
pip uninstall stable-diffusion-sdkit -y
pip install -r requirements_model_openclip.txt
sh ./generate_descriptors.sh ../../heisenberg/Vision_transformer_features/test_data/ ./descriptors ./experiments_open_clip.txt

## FOR PREM:
#pip uninstall open-clip-torch -y
#pip install -r requirements_model_sd_dino.txt
#sh ./generate_descriptors.sh ./test_data/ ./test_data/descriptors ./experiments_sd_dino.txt
#pip uninstall stable-diffusion-sdkit -y
#pip install -r requirements_model_openclip.txt
#sh ./generate_descriptors.sh ./test_data/ ./test_data/descriptors ./experiments_open_clip.txt