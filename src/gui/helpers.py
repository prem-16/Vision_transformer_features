import os


def get_image_list(image_directory):
    """
    Recursively get all image files in the directory and subdirectories.
    :return:
    """
    image_files = []
    image_dirs = []
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_files.append(file)
                image_dirs.append(os.path.join(root, file))
    return image_files, image_dirs