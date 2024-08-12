import matplotlib.pyplot as plt
import os

import numpy as np
from PIL import Image

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))

def absolute_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    
    return np.sum(np.abs(data - query), axis=axis_batch_size)

def compute_rates(score_method, query, data):
    match score_method:
        case 'absolute_difference':
            return absolute_difference(query, data)
        case 'mean_square_difference':
            return mean_square_difference(query, data)
        case 'cosine_similarity':
            return cosine_similarity(query, data)
        case 'correlation_cofficicient':
            return correlation_cofficicient(query, data)
        case _:
            print('Score Method isn\'t matched')
            return []
        
def correlation_cofficicient(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_mean = query - np.mean(query)
    data_mean = data - np.mean(data, axis=axis_batch_size, keepdims=True)
    query_norm = np.sqrt(np.sum(query_mean**2))
    data_norm = np.sqrt(np.sum(data_mean**2, axis=axis_batch_size))

    return np.sum(data_mean * query_mean, axis=axis_batch_size) / (data_norm * query_norm + np.finfo(float).eps)

def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
    result = np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)
    
    return result

def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)

    images_path = np.array(images_path)

    return images_np, images_path

def get_score(root_img_path, query_path, size, score_method):
    query = read_image_from_path(query_path, size)
    ls_path_score = []

    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            rates = compute_rates(score_method, query, images_np)
            if len(rates) == 0: continue

            ls_path_score.extend(list(zip(images_path, rates)))

    return query, ls_path_score

def get_correlation_coefficient_score(root_img_path, query_path, size):
    return get_score(root_img_path, query_path, size, 'correlation_cofficicient')

def get_cosine_similarity_score(root_img_path, query_path, size):
    return get_score(root_img_path, query_path, size, 'cosine_similarity')

def get_l1_score(root_img_path, query_path, size):
    return get_score(root_img_path, query_path, size, 'absolute_difference')

def get_l2_score(root_img_path, query_path, size):
    return get_score(root_img_path, query_path, size, 'mean_square_difference')

def mean_square_difference(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))

    return np.mean((data - query)**2, axis=axis_batch_size)

def plot_results(query_path, ls_path_score, reverse):
    fig = plt.figure(figsize=(10, 8))
    rows, cols = 4, 2

    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title('Query Image')
    plt.axis('off')
    plt.imshow(Image.open(query_path))

    first_5_matched_images = sorted_images_score(ls_path_score, reverse)[:5]

    for i, (img_path, score) in enumerate(first_5_matched_images):
        ax = fig.add_subplot(rows, cols, i + 3)
        ax.set_title(f'Top {i + 1}: {score}')
        plt.axis('off')
        plt.imshow(Image.open(img_path))

    plt.show(block=False)
    plt.pause(3)
    plt.close()

def read_image_from_path(path, size):
    im = Image.open(path).convert("RGB").resize(size)

    return np.array(im)

def sorted_images_score(ls_path_score, reverse):
    ls_path_score = np.array(ls_path_score, dtype=([('image_path', 'U100'), ('score', 'float32')]))
    sorted_path_score = np.sort(ls_path_score, order='score')

    if reverse:
        sorted_path_score = sorted_path_score[::-1]

    return sorted_path_score

def main():
    root_img_path = f"{ROOT}/train/"
    size = (448, 448)
    # L1 score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_l1_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_l1_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    # L2 score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_l2_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_l2_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)

    # Cosine similarity score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=True)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=True)

    # Correlation Coefficient
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=True)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=True)

if __name__ == "__main__":
    main()
