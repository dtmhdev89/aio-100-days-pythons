import cv2
import numpy as np
from concurrent import futures

from app_pixel_wise_matching import l1_distance, l2_distance, perform_save_result, compute_cost

def compute_cost_vector(cost_method, v1, v2):
    match cost_method:
        case 'l1_distance':
            return np.sum(l1_distance(v1, v2))
        case 'l2_distance':
            return np.sum(l2_distance(v1, v2))
        case 'cosine_similarity':
            return cosine_similarity_score(v1, v2)
        case _:
            raise f'unsupprted cost method {cost_method}'

def cosine_similarity_score(v1, v2):
    v1_norm = np.sqrt(np.sum(v1**2))
    v2_norm = np.sqrt(np.sum(v2**2))

    return (np.dot(v1, v2) / (v1_norm * v2_norm + np.finfo(np.float32).eps))

def window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size, cost_method, save_result=True):
    left_img = cv2.imread(left_img_path, flags=0)
    right_img = cv2.imread(right_img_path, flags=0)

    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)

    height, width = left_img.shape[:2]
    kernel_half = int((kernel_size - 1) // 2)
    max_value = 255 * 9
    scale = 3 # Could be another value

    depth = np.zeros((height, width), dtype=np.uint8)

    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            cost_min = 65534

            for j in range(disparity_range):
                total = 0
                element_value = 0

                for v in range(-kernel_half, kernel_half + 1):
                    for u in range(-kernel_half, kernel_half + 1):
                        element_value = max_value

                        if (x + u - j) >= 0:
                            element_value = compute_cost(cost_method, int(left_img[y + v, x + u]), int(right_img[y + v, x + u - j]))
                        
                        total += element_value
                    
                    if total < cost_min:
                        cost_min = total
                        disparity = j
            
            depth[y, x] = disparity * scale
    
    if save_result:
        print('Saving...')
        perform_save_result('window_based', cost_method, depth)
        print('Done')
    
    return depth

def window_based_matching_vector_version(left_img_path, right_img_path, disparity_range, kernel_size, cost_method, save_result=True):
    left_img = cv2.imread(left_img_path, flags=0)
    right_img = cv2.imread(right_img_path, flags=0)

    left_img = left_img.astype(np.float32)
    right_img = right_img.astype(np.float32)

    height, width = left_img.shape[:2]

    kernel_half = int((kernel_size - 1) // 2)
    scale = 3
    # max_value = 255 * 9
    cost_min = 65534

    costs = np.full((height, width, disparity_range), cost_min, dtype=np.float32)

    for y in range(kernel_half, height-kernel_half):
        for x in range(kernel_half, width-kernel_half):
            for j in range(disparity_range):
                d = x - j
                if cost_method == 'cosine_similarity':
                    cost = -1
                else:
                    cost = cost_min
                
                if (d - kernel_half) >= 0:
                    wp = left_img[(y-kernel_half):(y+kernel_half)+1, (x-kernel_half):(x+kernel_half)+1]
                    wqd = right_img[(y-kernel_half):(y+kernel_half)+1, (d-kernel_half):(d+kernel_half)+1]

                    wp_flattened = wp.flatten()
                    wqd_flattened = wqd.flatten()

                    cost = compute_cost_vector(cost_method, wp_flattened, wqd_flattened)

                costs[y, x, j] = cost

    if cost_method == 'cosine_similarity':
        min_cost_indices = np.argmax(costs, axis=-1)
    else:
        min_cost_indices = np.argmin(costs, axis=-1)

    depth = min_cost_indices * scale
    depth = depth.astype(np.uint8)

    if save_result:
        print('Saving..')
        perform_save_result('window_based_vector', cost_method, depth)
        print('Done')
    
    return depth

def main():
    # problem 1
    left_img_path = 'Aloe/Aloe_left_1.png'
    right_img_path = 'Aloe/Aloe_right_1.png'

    disparity_range = 64
    kernel_size = 3

    # window_based_matching_l1_result = window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size, 'l1_distance')
    # window_based_matching_l2_result = window_based_matching(left_img_path, right_img_path, disparity_range, kernel_size, 'l2_distance')

    # problem 2
    # window_based_matching_l1_vector_result = window_based_matching_vector_version(left_img_path, right_img_path, disparity_range, kernel_size, 'l1_distance')
    # window_based_matching_l2_vector_result = window_based_matching_vector_version(left_img_path, right_img_path, disparity_range, kernel_size, 'l2_distance')

    # problem 3
    # kernel_size = 5
    # right_img_path = 'Aloe/Aloe_right_2.png'
    # window_based_matching_l1_vector_result = window_based_matching_vector_version(left_img_path, right_img_path, disparity_range, kernel_size, 'l1_distance')

    # problem 4
    # window_based_matching_cosine_vector_result = window_based_matching_vector_version(left_img_path, right_img_path, disparity_range, kernel_size, 'cosine_similarity')

    with futures.ProcessPoolExecutor(max_workers=5) as executor:
        executor.submit(window_based_matching, left_img_path, right_img_path, disparity_range, kernel_size, 'l1_distance')

        executor.submit(window_based_matching_vector_version, left_img_path, right_img_path, disparity_range, kernel_size, 'l1_distance')

        kernel_size = 5
        right_img_path = 'Aloe/Aloe_right_2.png'

        executor.submit(window_based_matching_vector_version, left_img_path, right_img_path, disparity_range, kernel_size, 'l1_distance')
        executor.submit(window_based_matching_vector_version, left_img_path, right_img_path, disparity_range, kernel_size, 'cosine_similarity')


if __name__ == "__main__":
    main()
