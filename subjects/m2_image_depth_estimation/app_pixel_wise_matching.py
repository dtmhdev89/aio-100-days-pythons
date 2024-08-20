import os
import gdown
from zipfile import ZipFile

import cv2
import numpy as np

def predownload_data():
    tsukuba_gid = "14gf8bcym_lTcvjZQmg8kwq3aXkENBxMQ"
    aloe_gid = "1wxmiUdqMciuTOs0ouKEISl8-iTVXdOWn"
    output_tsukuba = 'tsukuba.zip'
    output_aloe = 'aloe.zip'
    if not os.path.isfile(output_tsukuba):
        gdown.download(id=tsukuba_gid, output=output_tsukuba)

    if not os.path.isfile(output_aloe):
        gdown.download(id=aloe_gid, output=output_aloe)
    
    for zipped_filename in [output_tsukuba, output_aloe]:
        if os.path.exists(os.path.join(zipped_filename.split('.')[0])):
            continue

        with ZipFile(os.path.join(zipped_filename), 'r') as zip_ref:
            for member in zip_ref.namelist()[1:]:
                if member:
                    zip_ref.extract(member, path=os.path.join(zipped_filename.split('.')[0]))

def compute_cost(cost_method, v1, v2):
    match cost_method:
        case 'l1_distance':
            return l1_distance(v1, v2)
        case 'l2_distance':
            return l2_distance(v1, v2)
        case _:
            raise f'Unsupported cost method {cost_method}'

def l1_distance(x, y):
    return abs(x - y)

def l2_distance(x, y):
    return (x - y)**2

def perform_save_result(algorithm_name, cost_method, depth):
    cv2.imwrite(f'{algorithm_name}_{cost_method}.png', depth)
    cv2.imwrite(f'{algorithm_name}_{cost_method}_color.png', cv2.applyColorMap(depth, cv2.COLORMAP_JET))

def pixel_wise_matching(left_img, right_img, disparity_range, cost_method, save_result=True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    # create blank disparity map
    depth = np.zeros((height, width), np.uint8)
    scale = 16
    max_value = 255

    for y in range(height):
        for x in range(width):
            disparity = 0
            cost_min = max_value

            for j in range(disparity_range):
                cost = max_value
                
                if (x - j) >= 0:
                    cost = compute_cost(cost_method, int(left[y, x]), int(right[y, x - j]))

                if cost < cost_min:
                    cost_min = cost
                    disparity = j
            
            # Let depth at (y, x) = j (disparity)
            # Multiple by a scale factor for visualization purpose
            # depth[y,x] = disparity * (255 / D) or any scale value that could be better.
            depth[y, x] = disparity * scale
    
    if save_result == True:
        print('Saving result...')
        perform_save_result('pixel_wise', cost_method, depth)

    print('Done')

    return depth

def pixel_wise_matching_vector_version(left_img, right_img, disparity_range, cost_method, save_result=True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    disparity_range = 16
    max_value = 255
    scale = round(255 / disparity_range)

    costs = np.full((height, width, disparity_range), max_value, dtype=np.float32)

    for d in range(disparity_range):
        left_d = left[:, d:width]
        right_d = right[:, 0:(width - d)]
        costs[:, d:width, d] = compute_cost(cost_method, left_d, right_d)

    min_cost_indices = np.argmin(costs, axis=-1)
    depth =  min_cost_indices * scale
    depth = depth.astype(np.uint8)

    if save_result == True:
        print('Saving result...')
        perform_save_result('pixel_wise_vector', cost_method, depth)

    return depth

def main():
    predownload_data()

    LEFT_IMG_PATH = 'tsukuba/left.png'
    RIGHT_IMG_PATH = 'tsukuba/right.png'

    disparity_range = 16
    pixel_wise_matching_l1_result = pixel_wise_matching(LEFT_IMG_PATH, RIGHT_IMG_PATH, disparity_range, 'l1_distance')
    pixel_wise_matching_l2_result = pixel_wise_matching(LEFT_IMG_PATH, RIGHT_IMG_PATH, disparity_range, 'l2_distance')

    pixel_wise_matching_l1_vector_result = pixel_wise_matching_vector_version(LEFT_IMG_PATH, RIGHT_IMG_PATH, disparity_range, 'l1_distance')
    pixel_wise_matching_l2_vector_result = pixel_wise_matching_vector_version(LEFT_IMG_PATH, RIGHT_IMG_PATH, disparity_range, 'l2_distance')



if __name__ == "__main__":
    main()
