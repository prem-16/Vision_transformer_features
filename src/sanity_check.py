import gzip
import pickle

descriptor_filename = "./test_data/descriptors/(id_1_6)_descriptor_OPEN_CLIP_data_rotation_X_episode_2.pkl.gzip"
# Load descriptors from pickle files
#   Extract gzip
f = gzip.open(descriptor_filename, 'rb')
#   Load pkl
pkl = pickle.load(f)

print(pkl.keys())
descriptors = pkl['descriptors'][0][0]
print(descriptors.shape)
num_patches = pkl['descriptors'][0][1]['num_patches'][0]
descriptor_size = descriptors.shape[-1]
print("num patches: ", num_patches)
print("descriptor size", descriptor_size)
descriptors_resized = descriptors.reshape((descriptors.shape[0], descriptors.shape[1], num_patches, num_patches, descriptor_size))
print("descriptors resized size", descriptors_resized.shape)
descriptors_resized_back = descriptors.reshape((descriptors.shape[0], descriptors.shape[1], num_patches * num_patches, descriptor_size))
print("descriptors resized back size", descriptors_resized_back.shape)
