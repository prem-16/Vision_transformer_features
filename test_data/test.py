# uncompress data.pkl.gzip and save to new folder called data_uncompressed

import gzip
import os
import pickle


# Uncompress the file
def uncompress_data(datasets_dir="./test_data"):
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)
    return data


# Save the uncompressed data
def save_uncompressed_data(data, datasets_dir="./test_data"):
    data_file = os.path.join(datasets_dir, 'data_uncompressed.pkl')
    f = open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()


if __name__ == "__main__":
    data = uncompress_data(datasets_dir="./test_data")
    save_uncompressed_data(data, datasets_dir="./test_data")

    # Open the pkl file and save as a folder
    # Path: test_data\data_uncompressed.pkl

    file = open("./test_data/data_uncompressed.pkl", 'rb')
    data = pickle.load(file)
    file.close()