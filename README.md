# Vision Transformer Features
## Setup
```
pip install requirements_dataset.txt
```
If using OpenCLIP:
```
pip install requirements_model_openclip.txt
```
Otherwise:
```
pip install requirements_model_sd_dino.txt
```

Removal of OpenCLIP for SD-DINO installation:
```
pip uninstall open-clip-torch
```
Removal of SD-DINO for OpenCLIP
```
pip uninstall stable-diffusion-sdkit
```

## ManiSkill2 Dataset Generation
Placeholder

## Descriptor Generation
### Descriptor generation for an individual episode.
An example command to generate descriptors for the sd-dino model:
```
python src/dataset_collection/generate_descriptors.py \
    --identifier 1_1 \
    --ignore_duplicates TRUE \
    --disable_timestamp TRUE \
    --dataset_path DATASET_PATH \
    --descriptor_dir OUTPUT_DIR \
    --load_size 448 \
    --model SD_DINO \
    --fuse_dino FALSE \
    --only_dino FALSE \
    --dino_v2 TRUE \
    --sd_load_size 960 \
    --pca FALSE \
    --raw TRUE \
    --raw_layer s4
```
Note: commands --fuse_dino --only_dino --dino_v2 --sd_load_size --pca -raw --raw_layer are model specific and will be defined in the corresponding wrapper file e.g. src/gui/models/sd_dino/sd_dino_wrapper.py.

### Batch descriptor generation for all episodes
Generation commands such as above can be batched into a single .txt file and run on all dataset data files.

E.g. in experiments_sd_dino.txt:
```
# Experiment 1_1: DINOv1
python src/dataset_collection/generate_descriptors.py --identifier 1_1 --ignore_duplicates TRUE --disable_timestamp TRUE --model SD_DINO --fuse_dino TRUE --only_dino TRUE --dino_v2 FALSE --load_size 448 --dataset_path $dataset_dir --descriptor_dir $output_dir --combine_pca FALSE --pca TRUE --stride 4
# Experiment 1_1_2: DINOv1
python src/dataset_collection/generate_descriptors.py --identifier 1_1_2 --ignore_duplicates TRUE --disable_timestamp TRUE --model SD_DINO --fuse_dino TRUE --only_dino TRUE --dino_v2 FALSE --load_size 448 --dataset_path $dataset_dir --descriptor_dir $output_dir --combine_pca FALSE --pca TRUE --stride 8
# Experiment 1_2: DINOv2
python src/dataset_collection/generate_descriptors.py --identifier 1_2 --ignore_duplicates TRUE --disable_timestamp TRUE --model SD_DINO --fuse_dino TRUE --only_dino TRUE --dino_v2 TRUE --load_size 448 --dataset_path $dataset_dir --descriptor_dir $output_dir --combine_pca FALSE --pca TRUE --stride 7
...
```

Then run:
```
sh ./generate_descriptors.sh ./test_data/ ./descriptors/ ./experiments_sd_dino.txt
```

## Descriptor evaluation
Given the generated descriptors (with the expected prefix `(id_$identifier)`), various configurations can be defined in get_performance.py in the configs dictionary.
For example:
```python
{
  # DINOv1 experiment with default
  "(id_1_1)": {"model_name": "SD_DINO", "exp_name": "DINOv1 - stride 4", "category": "DINOv1"},
  "(id_3_1)": {
      "model_name": "SD_DINO",
      "descriptor_config_ids": ["(id_1_1)"],
      "metric": separate_head_similarity(metric="cosine", head_size=6),
      "exp_name": "DINOv1 - stride 4, per-head cosine similarity",
      "category": "DINOv1"
  },
  "(id_3_2)": {
      "model_name": "OPEN_CLIP",
      "descriptor_config_ids": ["(id_1_6)"]
      "exp_name": "OpenCLIP",
      "category": "OpenCLIP"
  },
  "(id_1_6_2)": {
      "model_name": "SD_DINO",
      "descriptor_config_ids": ["(id_1_6)", "(id_1_3_4)"],
      "exp_name": "SD + OpenCLIP - s5 only ", "category": "SD_COMB",
  },
}
```

To evaluate these descriptors we run
```
python src/gui/get_performance.py
```

There is an optional argument `--filter_config` to only evaluate specific configurations e.g.
```
python src/gui/get_performance.py --filter_config "(id_3_1),(id_3_2)"
```

## Graph visualization
Currently three types of plots are available from get_plot.py and can be called as follows:
```python
transformations = [
    "rotation_X",
    "rotation_Y",
    "rotation_Z",
    "translation_X",
    "translation_Y",
    "translation_Z"
]
config_ids = ['(id_1_1)', '(id_1_2)', '(id_1_2_2)', '(id_1_2_3)', '(id_2_3)', '(id_1_6)']

# Plot mean config heatmap/distance error over all episodes for given transformation
plot_per_transform(
    config_ids, transformations, apply_log=False,
    apply_moving_avg=False, std_scale=0.5, log_std_scale=0.1, img_size=255
)

# Plot mean config heatmap/distance error over all episodes, poses/frames
# where the x-axis is the transformations.
plot_scatter(
    config_ids, transformations, apply_log=False,
    apply_moving_avg=False, std_scale=0.5, log_std_scale=0.1, img_size=255
)

# Plot mean config heatmap/distance error over all transformations, episodes, and poses.
plot_bar(
    config_ids, transformations, apply_log=False,
    apply_moving_avg=False, std_scale=0.5, log_std_scale=0.1, img_size=255
)
```

## GUI Descriptor correspondences visualization
To run the GUI application:
```
python src/gui/main.py
```
![Alt text](/imags/gui_screenshot.png?raw=true)
