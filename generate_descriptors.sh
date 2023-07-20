# Given arguments dataset_dir and output_dir, this script will call
# the line python src/dataset_collection/generate_descriptors.py --model SD_DINO --only_dino FALSE --fuse_dino 0 --load_size 518 --dataset_path {dataset_dir}/*.pkl.gzip --descriptor_dir ./test_data/descriptors/ --combine_pca=FALSE --pca=TRUE
# for each file *.pkl.gzip in dataset_dir.

# Note:
# Make sure to add a newline at the end of experiments.txt otherwise the last line will not be read!!!
# Each line in experiments.txt should be a single model configuration
# Below we apply this to every dataset.

# Example usage:
# bash ./generate_descriptors.sh ./test_data ./descriptors

# Dataset and output directory
dataset_dir=$1
output_dir=$2
# Experiments file
experiments_file=$3

# If experiments file is experiments_open_clip.txt
if [ "$experiments_file" = "experiments_open_clip.txt" ]; then
    pip uninstall stable-diffusion-sdkit -y
    pip install -r requirements_model_openclip.txt
fi
# If experiments file is experiments_sd_dino.txt
if [ "$experiments_file" = "experiments_sd_dino.txt" ]; then
    pip uninstall open-clip-torch -y
    pip install -r requirements_model_sd_dino.txt
fi

# Activate conda environment and set PYTHONPATH
source ~/anaconda3/etc/profile.d/conda.sh

conda activate VTF
export PYTHONPATH="${PYTHONPATH}:./"

for f in $dataset_dir/*.pkl.gzip; do
    echo "Processing $f file..."
    # Run each line in experiments.txt
    while IFS= read -r line; do
        echo "$line"
        eval $line
    done < "$experiments_file"
done