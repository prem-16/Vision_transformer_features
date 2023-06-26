import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
from PIL import Image
import cv2
def read_data(datasets_dir="./test_data", index=0):
    """
    This method reads the states from the data file(data.pkl.gzip).
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file, 'rb')
    data = pickle.load(f)
    print(len(data['image_rgb']))
    # get images as features and actions as targets
    img = np.array(data['image_rgb'][index])
    img = Image.fromarray(img)
    sample = {
        "image_rgb": data['image_rgb'][index],
        "extrinsic": data['extrinsic'][index],
        "intrinsic": data['intrinsic'][index],
        "depth": data['depth'][index],
    }
    return sample, img

def transform_points(image1_data, image2_data , image1_points):
    """
    Transform the points from image 1 to image 2.
    :param image1_data: The data of image 1.
    :param image2_data: The data of image 2.
    :param image1_points: The points in image 1.
    :return: The points in image 2.
    """

    # # Transform the points
    depth = float( image1_data['depth'][image1_points[1]][image1_points[0]])
    image1_points_h = np.array([image1_points[0]*depth, image1_points[1]*depth, depth])
    image1_points_camera = np.matmul(np.linalg.inv(image1_data['intrinsic']), image1_points_h)
    image1_points_camera_h = np.append( image1_points_camera, 1)
    image1_points_world_h = np.matmul(np.linalg.inv(image1_data['extrinsic']), image1_points_camera_h)

    image2_points_camera_h = np.matmul(image2_data['extrinsic'], image1_points_world_h)
    image2_points_camera = image2_points_camera_h[:3] / image2_points_camera_h[3]
    image2_points_h = np.matmul(image2_data['intrinsic'], image2_points_camera)
    image2_points = image2_points_h[:2] / image2_points_h[2]

   
    return image2_points

def create_ground_truth_map(image_points , image):
    gt_map = np.zeros(image.shape)
    for image_point in image_points:
        gt_map[int(image_point[1])][int(image_point[0])] = 1
    return gt_map

if __name__ == "__main__":

    image1_data, img1 = read_data(datasets_dir="./test_data", index=0)

    r = cv2.selectROI("select the object", image1_data['image_rgb'])
    cv2.destroyAllWindows()
    image1_points = np.array([[r[0], r[1]],
                              [r[0] + r[2], r[1]],
                              [r[0] + r[2], r[1] + r[3]],
                              [r[0], r[1] + r[3]]])

    image2_data, img2 = read_data(datasets_dir="./test_data", index=2)
    image2_points = np.zeros((4, 2))
    for i,image1_point in enumerate(image1_points):

        image2_points[i] = transform_points(image1_data, image2_data, image1_point)

    plt.imshow(image1_data['image_rgb'])
    print(image1_points[:, 0])
    plt.scatter(image1_points[:, 0], image1_points[:, 1], c='r', marker='x')
    plt.show()
    gt_map_image2 = create_ground_truth_map(image2_points, image2_data['image_rgb'])
    plt.imshow(image2_data['image_rgb'])
    plt.scatter(image2_points[:, 0], image2_points[:, 1], c='r', marker='x')
    plt.show()
