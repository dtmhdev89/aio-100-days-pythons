import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
import chromadb

from basic_retrieval import CLASS_NAME, ROOT
from basic_retrieval import folder_to_images, read_image_from_path, compute_rates
from advance_retrieval_with_pretrained_model import get_single_image_embedding

def add_embedding(collection, files_path, embedding_function):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = np.array(Image.open(filepath))
        embedding = get_single_image_embedding(embedding_function, image)
        embeddings.append(embedding.tolist())
    
    collection.add(embeddings=embeddings, ids=ids)

def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + "/" + filename
            files_path.append(filepath)
    
    return files_path

def plot_results(query_path, files_path, results):
    fig = plt.figure(figsize=(10, 8))
    rows, cols = 4, 2

    ax = fig.add_subplot(rows, cols, 1)
    ax.set_title('Query Image')
    plt.axis('off')
    plt.imshow(Image.open(query_path))

    queried_ids = results.get('ids')
    queried_ids = [int(str_id.split('_')[-1]) for str_id in queried_ids[0]]

    for i, id in enumerate(queried_ids):
        ax = fig.add_subplot(rows, cols, i + 3)
        ax.set_title(f'Top {i + 1}:')
        plt.axis('off')
        plt.imshow(Image.open(files_path[id]))

    plt.show(block=False)
    plt.pause(7)
    plt.close()
    pass

def search(image_path, collection, n_results, embedding_function):
    query_image = np.array(Image.open(image_path))
    query_embedding = get_single_image_embedding(embedding_function, query_image)
    results = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results=n_results
    )

    return results

def main():
    embedding_function = OpenCLIPEmbeddingFunction()
    data_path = f'{ROOT}/train'
    test_path = f'{ROOT}/test'
    files_path = get_files_path(data_path)
    test_files_path = get_files_path(test_path)

    chroma_client = chromadb.Client()

    # L2 Score
    collection_name = "l2_collection"
    collections = chroma_client.list_collections()
    if collections.count(collection_name) > 0:
        chroma_client.delete_collection(name=collection_name)

    l2_collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "l2"})
    add_embedding(collection=l2_collection, files_path=files_path, embedding_function=embedding_function)

    test_img_path = test_files_path[1]
    l2_results = search(image_path=test_img_path, collection=l2_collection, n_results=5, embedding_function=embedding_function)
    plot_results(query_path=test_img_path, files_path=files_path, results=l2_results)

    test_img_path = test_files_path[2]
    l2_results = search(image_path=test_img_path, collection=l2_collection, n_results=5, embedding_function=embedding_function)
    plot_results(query_path=test_img_path, files_path=files_path, results=l2_results)

    # Cosine Similarity Collection
    collection_name = "cosine_similarity_collection"
    collections = chroma_client.list_collections()
    if collections.count(collection_name) > 0:
        chroma_client.delete_collection(name=collection_name)

    cosine_collection = chroma_client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": 'cosine'})
    add_embedding(collection=cosine_collection, files_path=files_path, embedding_function=embedding_function)

    test_img_path = test_files_path[1]
    cosine_results = search(image_path=test_img_path, collection=cosine_collection, n_results=5, embedding_function=embedding_function)
    plot_results(query_path=test_img_path, files_path=files_path, results=cosine_results)

    test_img_path = test_files_path[2]
    cosine_results = search(image_path=test_img_path, collection=cosine_collection, n_results=5, embedding_function=embedding_function)
    plot_results(query_path=test_img_path, files_path=files_path, results=cosine_results)

if __name__ == "__main__":
    main()
