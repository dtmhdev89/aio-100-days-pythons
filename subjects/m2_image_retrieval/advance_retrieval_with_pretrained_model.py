import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

from basic_retrieval import folder_to_images, read_image_from_path, compute_rates, plot_results

ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))

def get_score(root_img_path, query_path, size, score_method, embedding_function):
    query = read_image_from_path(query_path, size)
    query_embedding = get_single_image_embedding(embedding_function, query)
    ls_path_score = []

    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size)
            embedding_list = []
            for idx_img in range(images_np.shape[0]):
                embedding = get_single_image_embedding(embedding_function, images_np[idx_img].astype(np.uint8))

                embedding_list.append(embedding)

            rates = compute_rates(score_method, query_embedding, np.stack(embedding_list))
            if len(rates) == 0: continue

            ls_path_score.extend(list(zip(images_path, rates)))

    return query, ls_path_score

def get_correlation_coefficient_score(root_img_path, query_path, size, embedding_function):
    return get_score(root_img_path, query_path, size, 'correlation_cofficicient', embedding_function)

def get_cosine_similarity_score(root_img_path, query_path, size, embedding_function):
    return get_score(root_img_path, query_path, size, 'cosine_similarity', embedding_function)

def get_l1_score(root_img_path, query_path, size, embedding_function):
    return get_score(root_img_path, query_path, size, 'absolute_difference', embedding_function)

def get_l2_score(root_img_path, query_path, size, embedding_function):
    return get_score(root_img_path, query_path, size, 'mean_square_difference', embedding_function)

def get_single_image_embedding(embedding_function, image):
    embedding = embedding_function._encode_image(image=image)

    return np.array(embedding)

def main():
    embedding_function = OpenCLIPEmbeddingFunction()
    root_img_path = f"{ROOT}/train/"
    size = (448, 448)

    # L1 score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_l1_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=False)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_l1_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=False)

    # # L2 score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_l2_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=False)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_l2_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=False)

    # Cosine Similarity Score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=True)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=True)

    # Correlation Coefficient Score
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    _query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=True)

    query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
    _query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size, embedding_function)
    plot_results(query_path, ls_path_score, reverse=True)

if __name__ == "__main__":
    main()
