import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as sk_r2_score

from dataset_man import dataset_manager

class CustomLinearRegression:
    def __init__(self, X_data, y_target, lr=1e-2, num_epochs=10000) -> None:
        self.num_samples = X_data.shape[0]
        self.X_data = np.c_[np.ones((self.num_samples, 1)), X_data]
        self.y_target = y_target
        self.lr = lr
        self.num_epochs = num_epochs

        self.theta = np.random.randn(self.X_data.shape[1], 1)
        self.losses = []

    def compute_loss(self, y_pred, y_target):
        loss = (y_pred - y_target)**2 / self.num_samples

        return loss
    
    def predict(self, X_data):
        y_pred = X_data.dot(self.theta)

        return y_pred
    
    def fit(self):
        for epoch in range(self.num_epochs):
            y_pred = self.predict(self.X_data)

            loss = self.compute_loss(y_pred, self.y_target)
            self.losses.append(loss)

            loss_grd = 2 * (y_pred - self.y_target) / self.num_samples
            gradients = self.X_data.T.dot(loss_grd)

            self.theta = self.theta - self.lr * gradients

            if (epoch % 50) == 0:
                print(f'Epoch: {epoch} - Loss: {loss}')

        return {
            'loss': sum(self.losses) / len(self.losses),
            'weight': self.theta
        }
    
def r2_score(y_pred, y):
    rss = np.sum((y - y_pred) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    score = 1 - (rss / tss)

    return score

def create_polynomial_features(X, degree=2):
    """
    Creates polynomial features of a given degree.

    Args:
        X: A numpy array of shape (n_samples, n_features).
        degree: The degree of the polynomial features.

    Returns:
        A numpy array of shape (n_samples, n_features * degree) containing the polynomial features.
    """

    # n_samples, n_features = X.shape
    # X_new = np.ones((n_samples, n_features * degree))

    # for i in range(1, degree + 1):
    #     X_new[:, i * n_features:(i + 1) * n_features] = np.power(X, i)

    X_new = np.array(X)
    for d in range(2, degree + 1):
        X_new = np.c_[X_new, np.power(X, d)]

    return X_new

def create_polynomial_features_v2(X, degree=2):
    """
    Creates polynomial features of a given degree.

    Args:
        X: A numpy array of shape (n_samples, n_features).
        degree: The degree of the polynomial features.

    Returns:
        A numpy array of shape (n_samples, n_features * degree) containing the polynomial features.
    """

    # n_samples, n_features = X.shape
    # X_new = np.ones((n_samples, n_features * degree))

    # for i in range(1, degree + 1):
    #     X_new[:, i * n_features:(i + 1) * n_features] = np.power(X, i)

    X_mem = []
    for X_sub in X.T:
        X_sub = X_sub.T
        X_new = X_sub
        for d in range(2, degree + 1):
            X_new = np.c_[X_new, np.power(X_sub, d)]

        X_mem.extend(X_new.T)

    return np.c_[X_mem].T

def main():
    indication_line = '-' * 4

    print(f"{indication_line}Q4:")

    y_pred = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 2, 3, 4, 5])
    print(r2_score(y_pred, y))

    y_pred = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 5, 2, 4])

    print(r2_score(y_pred, y))

    print(f"{indication_line}Q7:")

    X = np.array([[1], [2], [3]])
    print(create_polynomial_features(X))

    print(f"{indication_line}Q8:")

    X = np.array([[1, 2], [2, 3], [3, 4]])
    degree = 2

    X_poly = create_polynomial_features_v2(X, degree)
    print(X_poly)

    print(f'{indication_line} Sales Predictions:')

    df = pd.DataFrame(dataset_manager.load_dataset('m4.SalesPrediction', as_dataframe=True))
    print(df.head())
    print(df.isna().sum())
    print(df.info())

    df = pd.get_dummies(df)
    print(df.head())

    print(df.mean())
    df = df.fillna(df.mean())

    feature_columns = ['TV', 'Radio', 'Social Media', 'Influencer_Macro', 'Influencer_Mega', 'Influencer_Micro', 'Influencer_Nano']
    label_column = ['Sales']

    X = df[feature_columns]
    y = df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    print(f'{indication_line}Q10:')

    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train)
    X_test_processed = scaler.transform(X_test)
    print(scaler.mean_[0])

    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train_processed)
    X_test_poly = poly_features.transform(X_test_processed)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    preds = poly_model.predict(X_test_poly)
    print(sk_r2_score(y_test, preds))

if __name__ == "__main__":
    main()
