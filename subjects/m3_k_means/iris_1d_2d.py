import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from dataset_man import DatasetManager

def main():
    dm = DatasetManager()
    print(dm.data_list)
    data = dm.load_dataset("m3.kmeans.20240830.iris_1D")
    X = np.array(data['Petal_Length'])
    print(len(X))

    X_reshape = X.reshape(-1, 1)

    # Simple Kmeans
    kmeans = KMeans(n_clusters=2, random_state=1)
    kmeans.fit(X_reshape)
    labels = kmeans.labels_
    print(kmeans.inertia_)

    for x, label in zip(X, labels):
        print(f"Cluster {label}: {x}")

    data = dm.load_dataset("m3.kmeans.20240830.iris_2D")
    X_df = pd.DataFrame(data)
    print(X_df.head())

    X = X_df[['Petal_Length', 'Petal_Width']].to_numpy()
    X_reshape = X.reshape(-1, 2)
    print(X_reshape)

    kmeans = KMeans(n_clusters=2, random_state=1)
    kmeans.fit(X_reshape)
    labels = kmeans.labels_
    print(kmeans.inertia_)

    for x, label in zip(X, labels):
        print(f"Cluster {label}: {x}")

    wcss_values = []
    for i in range(1, 6):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X.reshape(-1, 2))
        wcss = kmeans.inertia_
        wcss_values.append(wcss)
        
    plt.plot(range(1, 6), wcss_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('WCSS vs. Number of Clusters')
    plt.show()

    # try to scale data to see if any difference
    scale = StandardScaler()
    X_std = scale.fit_transform(X.astype(float))

    print(X_std)

    kmeans = KMeans(n_clusters=2, random_state=1)
    kmeans.fit(X_std)
    labels = kmeans.labels_
    print(kmeans.inertia_)

    for x, label in zip(X, labels):
        print(f"Cluster {label}: {x}")

    wcss_values = []
    for i in range(1, 6):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X.reshape(-1, 2))
        wcss = kmeans.inertia_
        wcss_values.append(wcss)
        
    plt.plot(range(1, 6), wcss_values)
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('WCSS vs. Number of Clusters')
    plt.show()

if __name__ == "__main__":
    main()
