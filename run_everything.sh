# Run  bash ./generate_descriptors.sh ./test_data ./descriptors ./experiments_open_clip.txt and bash ./generate_descriptors.sh ./test_data ./descriptors ./experiments_sd_dino.txt
pip uninstall open-clip-torch -y
pip install -r requirements_model_sd_dino.txt
sh ./generate_descriptors.sh ./test_data/ ./descriptors ./experiments_sd_dino.txt
pip uninstall stable-diffusion-sdkit -y
pip install -r requirements_model_openclip.txt
./generate_descriptors.sh ./test_data/ ./descriptors ./experiments_open_clip.txt