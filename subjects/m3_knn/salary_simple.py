import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

from dataset_man import DatasetManager

def main():
    dm = DatasetManager()
    print(dm.data_list)

    df = pd.DataFrame(dm.load_dataset("m3.knn.20240827.Salary_Data_simple"))

    print(df.head())

    sns.scatterplot(df, x='Salary', y='Experience')
    plt.show()

    sns.boxplot(df[['Experience']])
    plt.show()

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1:].to_numpy()

    print(type(X))

    knn_rg = KNeighborsRegressor(n_neighbors=3)
    knn_rg.fit(X, y)

    x_test = 4.3
    predicted_salary = knn_rg.predict([[x_test]])
    print(predicted_salary)

if __name__ == "__main__":
    main()
