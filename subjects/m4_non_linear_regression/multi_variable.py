import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from dataset_man import dataset_manager

def r2_score(y_pred, y):
    rss = np.sum((y_pred - y)**2)
    tss = np.sum((y - y.mean())**2)

    r2 = 1 - (rss / tss)

    return r2

class LinearRegression:
    def __init__(self, X_data, y_target, learning_rate=1e-5, num_epochs=10000):
        self.X_data = X_data
        self.y_target = y_target
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_samples = self.X_data.shape[0]
        self.theta = np.random.randn(self.X_data.shape[1])
        self.losses = []

    def compute_loss(self, y_pred, y_target):
        loss = (y_pred-y_target)*(y_pred-y_target)
        loss = np.mean(loss)
        return loss

    def predict(self, X_data):
        y_pred = X_data.dot(self.theta)
        return y_pred

    def fit(self):
        for epoch in range(self.num_epochs):
            y_pred = self.predict(self.X_data)

            loss = self.compute_loss(y_pred, self.y_target)
            self.losses.append(loss)

            k = 2*(y_pred-self.y_target)
            gradients = self.X_data.T.dot(k)/self.num_samples

            self.theta = self.theta - self.learning_rate*gradients

            # print(f'Epoch: {epoch} - Loss: {loss}')

        return {
            'loss': sum(self.losses)/len(self.losses),
            'weight': self.theta
        }
    
def create_polynomial_features(X, degree=2):
    """Creates the polynomial features
    Args:
        X: An array for the data.
        degree: An integer for the degree of
        the generated polynomial function.
    """
    X_mem = []
    for X_sub in X.T:
        X_sub = X_sub.T
        X_new = X_sub
        for d in range(2, degree+1):
            X_new = np.c_[X_new, np.power(X_sub, d)]

        X_mem.extend(X_new.T)

    return np.c_[X_mem].T

def main():
    df = pd.DataFrame(dataset_manager.load_dataset('m4.Fish', as_dataframe=True))

    print(df.head())
    print(df.info())

    # One-hot encode
    encode_species = pd.get_dummies(df.Species)
    print(type(encode_species), '\n', encode_species.head())

    # Label encode
    df['Species'] = df['Species'].astype('category')
    label_encoding_species = df['Species'].cat.codes
    print(type(label_encoding_species), '\n', label_encoding_species.unique())

    new_df = pd.concat([df, encode_species], axis='columns')
    print(new_df.head())

    X = new_df[[
        'VerticalLen', 'DiagonalLen', 'CrossLen', 'Height', 'Width',
        'Bream', 'Parkki', 'Perch', 'Pike', 'Roach', 'Smelt', 'Whitefish'
    ]]
    y = new_df['Weight']

    X, y = X.values, y.values
    X_data = np.hstack([np.ones((X.shape[0], 1)), X])
    print(X_data[:5])

    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
    print('X train shape: ', X_train.shape)
    print('X test shape: ', X_test.shape)
    print('y train shape: ', y_train.shape)
    print('y test shape: ', y_test.shape)

    linear_model = LinearRegression(X_train, y_train, learning_rate=1e-5, num_epochs=1000)
    print(linear_model.fit())

    preds = linear_model.predict(X_train)
    r2_train = r2_score(preds, y_train)

    preds = linear_model.predict(X_test)
    r2_test = r2_score(preds, y_test)

    print(r2_train, r2_test)

    # Polynomial Regression
    ## Simple Approach Form (a + b)^2 => a^2 + b^2 + a + b + 1

    X_new = np.array([[1, 2], [3, 4]])
    print(X_new)

    results = create_polynomial_features(X_new, degree=2)
    print(results)

    X_poly = create_polynomial_features(X, degree=2)
    X_data = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
    poly_model = LinearRegression(X_train, y_train, learning_rate=0.0000001, num_epochs=100000)
    poly_model.fit()

    preds = poly_model.predict(X_train)
    print(r2_score(preds, y_train))

    preds = poly_model.predict(X_test)
    print(r2_score(preds, y_test))

    # With sklearn
    poly_features = PolynomialFeatures(degree=2)
    X_data = poly_features.fit_transform(X)
    print(X_data[0])
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
    poly_model = LinearRegression(X_train, y_train, learning_rate=0.0000001, num_epochs=100000)
    poly_model.fit()

    preds = poly_model.predict(X_train)
    print(r2_score(preds, y_train))

if __name__ == "__main__":
    main()
