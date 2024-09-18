import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor

def main():
    X = np.array([3, 5, 8, 10, 12])
    y = np.array([12, 20, 28, 32, 36])

    X = X.reshape(-1, 1).astype('float')
    y = y.reshape(-1, 1).astype('float')

    print(X)
    print(y)
    decision_tree_regressor = DecisionTreeRegressor(random_state=0, criterion="squared_error")
    decision_tree_regressor.fit(X, y)
    fig, ax = plt.subplots(figsize=(12, 8))
    tree.plot_tree(decision_tree_regressor, ax=ax, filled=True)

    plt.show()

if __name__ == "__main__":
    main()
