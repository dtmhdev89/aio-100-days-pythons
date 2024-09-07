import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dataset_man import DatasetManager

def main():
    dm = DatasetManager()
    print(dm.data_list)
    data = pd.DataFrame(dm.load_dataset("m3.knn.20240827.iris_2D"))
    # get x
    x_data = data[['Petal_Length', 'Petal_Width']].to_numpy()
    x_data = x_data.reshape(6, 2)
    print(x_data)

    # # get y
    y_data = data['Label'].to_numpy()
    print(y_data)

    # get model
    classifier = KNeighborsClassifier(n_neighbors=6)
    classifier.fit(x_data, y_data)

    # unknown input
    x_test = [[2.6, 0.7]]
    y_pred = classifier.kneighbors(x_test)
    print(y_pred)

    # in mm (milimet unit)
    data = pd.DataFrame(dm.load_dataset("m3.knn.20240827.iris_2D_mm"))

    # get x
    x_data = data[['Petal_Length', 'Petal_Width']].to_numpy()
    x_data = x_data.reshape(6, 2)
    print(x_data)

    # get y
    y_data = data['Label'].to_numpy()
    print(y_data)

    classifier = KNeighborsClassifier(n_neighbors=6)
    classifier.fit(x_data, y_data)

    # unknown input
    x_test = [[2.6, 7.0]]
    y_pred = classifier.kneighbors(x_test)
    print(y_pred)

    # using scale
    data = pd.DataFrame(dm.load_dataset("m3.knn.20240827.iris_2D"))
    # get x
    x_data = data[['Petal_Length', 'Petal_Width']].to_numpy()
    x_data = x_data.reshape(6, 2)
    print(x_data)

    y_data = data['Label'].to_numpy()

    scale = StandardScaler()
    x_std = scale.fit_transform(x_data.astype(float))

    print('use scaling on 2d')
    print(x_std)

    classifier = KNeighborsClassifier(n_neighbors=6)
    classifier.fit(x_std, y_data)

    # unknown input
    x_test = [[2.6, 0.7]]
    x_test_std = scale.transform(x_test)
    y_pred = classifier.kneighbors(x_test_std)
    print(y_pred)

    # in mm (milimet unit)
    data = pd.DataFrame(dm.load_dataset("m3.knn.20240827.iris_2D_mm"))

    # get x
    x_data = data[['Petal_Length', 'Petal_Width']].to_numpy()
    x_data = x_data.reshape(6, 2)
    print(x_data)

    # get y
    y_data = data['Label'].to_numpy()

    scale = StandardScaler()
    x_std = scale.fit_transform(x_data)

    print(x_std)

    classifier = KNeighborsClassifier(n_neighbors=6)
    classifier.fit(x_std, y_data)

    x_test = [[2.6, 7.0]]
    x_test_std = scale.transform(x_test)

    y_pred = classifier.kneighbors(x_test_std)

    print(y_pred)

if __name__ == "__main__":
    main()
