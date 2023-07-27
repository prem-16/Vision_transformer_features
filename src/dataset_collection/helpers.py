import gzip
import os
import pickle


def store_data(data, datasets_dir="./test_data", descriptor_name=None):
    if descriptor_name is None:
        descriptor_name = 'data_rotation_apple.pkl.gzip'
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, descriptor_name)
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()
