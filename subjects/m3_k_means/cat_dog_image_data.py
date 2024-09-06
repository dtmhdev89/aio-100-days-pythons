import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from transformers import CLIPProcessor, CLIPModel
import torch

def main():
    def load_images_from_folder(folder):
        data = []
        origin_labels = []
        for filename in os.listdir(folder):
            if filename.lower().endswith('.jpg'):
                img = cv2.imread(os.path.join(folder, filename))
                data.append(img)
                origin_labels.append(filename.lower())

        return origin_labels, data
    
    dog_labels, dog_images = load_images_from_folder('cat_dog_dataset/dog')
    cat_labels, cat_images = load_images_from_folder('cat_dog_dataset/cat')

    origin_labels = np.concatenate((dog_labels, cat_labels), axis=0)
    print(origin_labels)

    print(type(dog_images))
    print(type(cat_images))

    dog_images = np.array(dog_images)
    cat_images = np.array(cat_images)

    print(dog_images.shape)
    print(cat_images.shape)

    dog_images_reshape = dog_images.reshape(10, -1)
    cat_images_reshape = cat_images.reshape(10, -1)
    print(dog_images_reshape.shape)
    print(cat_images_reshape.shape)

    merged_features = np.concatenate((dog_images_reshape, cat_images_reshape), axis=0)
    # like using vstack
    # merged_features = np.vstack((dog_images_reshape, cat_images_reshape))
    print(merged_features.shape)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(merged_features)
    labels = kmeans.labels_
    silh_score = silhouette_score(merged_features, labels)
    for label, cluster_label in zip(origin_labels, labels):
        print('cluster ', cluster_label, ': ', label)

    print(silh_score)

    # Mặc dù kết quả là khá tốt, tuy nhiên nếu xét theo độ đo silhouette_score thì khá gần 0 (0.038)
    # Khả năng phân nhóm bị sai/trùng lắp từ nhóm này qua nhóm kia.

    silh_scores = []
    wcss_values = []
    for i in range(2, 6):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(merged_features)
        wcss_values.append(kmeans.inertia_)
        silh_scores.append(silhouette_score(merged_features, kmeans.labels_))
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 6), silh_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 6), wcss_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('WCSS for Optimal K')
    plt.show()

    # Use magic function

    def magic_function(images):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        inputs = clip_processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)

        return features.numpy()
    
    dog_images_features = magic_function(dog_images)
    cat_images_features = magic_function(cat_images)

    print(dog_images_features.shape)
    print(cat_images_features.shape)

    # merged_features = np.concatenate((dog_images_features, cat_images_features), axis=0)
    merged_features = np.vstack((dog_images_features, cat_images_features))
    print(merged_features.shape)

    kmeans = KMeans(n_clusters=2, random_state=0).fit(merged_features)
    print(kmeans.labels_)
    silh_score = silhouette_score(merged_features, kmeans.labels_)
    print(silh_score)
    print(kmeans.inertia_)

    for real_label, predicted_label in zip(origin_labels, kmeans.labels_):
        print(f"Cluster {predicted_label}: {real_label}")

    silh_scores = []
    wcss_values = []
    for i in range(2, 6):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(merged_features)
        wcss_values.append(kmeans.inertia_)
        silh_scores.append(silhouette_score(merged_features, kmeans.labels_))
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 6), silh_scores, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal K')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 6), wcss_values, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS')
    plt.title('WCSS for Optimal K')
    plt.show()

if __name__ == "__main__":
    main()
