import numpy as np
import cv2

def get_flipping_matrix(type):
    return {
        "v":
            np.array([
                [1, 0],
                [0, -1]
            ]),
        "h":
            np.array([
                [-1, 0],
                [0, 1]
            ])
    }[type]

def compute_padding_matrix(type, img):
    try:
        height, width = img.shape
    except:
        height, width, _depth = img.shape

    return {
        "v": np.array([0, height - 1]),
        "h": np.array([width - 1, 0])
    }[type]

def grayscale_flip(type, img):
    height, width = img.shape
    x1_np = np.arange(width)
    x2_np = np.arange(height)
    combine_w_h_matrix = np.array([[x1, x2] for x2 in x2_np for x1 in x1_np])
    map_color = lambda arr: img[arr[1], arr[0]]
    # color = np.apply_along_axis(map_color, axis=1, arr=combine_w_h_matrix)
    # to_origin = color.reshape(width, height).T
    flip_matrix = get_flipping_matrix(type)
    padding_matrix = compute_padding_matrix(type, img)
    new_combine_w_h_matrix = np.add(flip_matrix.dot(combine_w_h_matrix.T).T, padding_matrix)
    new_img = np.apply_along_axis(map_color, axis=1, arr=new_combine_w_h_matrix)
    new_img = new_img.reshape(height, width)
    cv2.imwrite(f'natural_gray_{type}.png', new_img)

def color_flip(type, img):
    height, width, depth = img.shape
    x1_np = np.arange(width)
    x2_np = np.arange(height)
    combine_w_h_matrix = np.array([[x1, x2] for x2 in x2_np for x1 in x1_np])
    map_color = lambda arr: img[arr[1], arr[0], :]
    # color = np.apply_along_axis(map_color, axis=1, arr=combine_w_h_matrix)
    # to_origin = color.reshape(width, height).T
    flip_matrix = get_flipping_matrix(type)
    padding_matrix = compute_padding_matrix(type, img)
    new_combine_w_h_matrix = np.add(flip_matrix.dot(combine_w_h_matrix.T).T, padding_matrix)
    new_img = np.apply_along_axis(map_color, axis=1, arr=new_combine_w_h_matrix)
    new_img = new_img.reshape(height, width, -1)
    cv2.imwrite(f'natural_{type}.png', new_img)

def main():
    img = cv2.imread('nature_gray.png', 0)
    grayscale_flip("v", img)
    grayscale_flip("h", img)

    color_img = cv2.imread('nature.png', 1)
    color_flip("v", color_img)
    color_flip("h", color_img)

if __name__ == "__main__":
    main()
