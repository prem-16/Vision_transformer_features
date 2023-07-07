import gzip
import os
import pickle


def store_data(data, datasets_dir="./test_data"):
    # save data
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'data_rotation_apple.pkl.gzip')
    f = gzip.open(data_file, 'wb')
    pickle.dump(data, f)
    f.close()
