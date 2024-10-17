import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from scipy import optimize

from dataset_man import dataset_manager

def r2_score(y_pred, y):
    rss = np.sum((y_pred - y) ** 2)
    tss = np.sum((y - y.mean()) ** 2)

    r2 = 1 - (rss / tss)

    return r2

class LinearRegression:
    def __init__(self, X_data, y_target, lr=1e-2, num_epochs=10000) -> None:
        self.X_data = X_data
        self.y_target = y_target
        self.lr = lr
        self.num_epochs = num_epochs
        self.n_samples = self.X_data.shape[0]

        self.theta =  np.random.randn(self.X_data.shape[1])
        self.losses = []

    def compute_loss(self, y_pred, y_target):
        loss = (y_pred - y_target) * (y_pred - y_target)
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

            k = 2 * (y_pred - self.y_target)
            gradients = self.X_data.T.dot(k) / self.n_samples

            self.theta = self.theta - self.lr * gradients

            print(f'Epoch: {epoch} - Loss: {loss}')

        return   {
            'loss': sum(self.losses) / len(self.losses),
            'weight': self.theta
        }
    
def create_polynomial_features(X, degree=2):
    """
    Creates the polynomial features
    Args:
        X: An array tensor for the data.
        degree: An integer for the degree of the generated polynomial function
    """
    X_new = X

    for d in range(2, degree + 1):
        X_new = np.c_[X_new, np.power(X, d)]

    return X_new

def func(x, a, b):
    y = a*np.exp(b*x)
    return y

def main():
    df = pd.DataFrame(dataset_manager.load_dataset('m3.decision.tree.regression.20240906.Position_Salaries', as_dataframe=True))
    print(df.head())

    X = df.Level
    y = df.Salary

    plt.scatter(X, y)
    plt.xlabel('Level')
    plt.ylabel('Salary')
    # plt.show()

    X_linear = X.values.reshape(-1, 1)
    X_linear = np.hstack((np.ones((X_linear.shape[0], 1)), X_linear))

    linear_model = LinearRegression(X_linear, y, num_epochs=100)
    linear_model.fit()

    y_pred = linear_model.predict(X_linear)
    r2 = r2_score(y_pred, y)
    print(r2)
    print(linear_model.theta)

    plt.plot(X, y, 'yo', X, linear_model.theta[1] * X + linear_model.theta[0], '--k')
    # plt.show()

    # Polymial Regression with degree = 2
    X_poly = create_polynomial_features(X, degree=2)
    print(X_poly[:5])

    X_poly = np.hstack((np.ones((X_poly.shape[0], 1)), X_poly))

    poly_model = LinearRegression(X_poly, y, lr=1e-4, num_epochs=10000)
    poly_model.fit()

    y_pred = poly_model.predict(X_poly)
    r2 = r2_score(y_pred, y)

    print(r2, poly_model.theta)

    X_plot = df.Level
    y_func = poly_model.theta[2]*(X**2) + poly_model.theta[1] * X + poly_model.theta[0]
    plt.plot(X_plot, y, 'yo', X, y_func, '--k', color='g')

    X_poly = create_polynomial_features(X, degree=3)
    X_poly = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
    print(X_poly)

    poly_model = LinearRegression(X_poly, y, lr=1e-6, num_epochs=500)
    poly_model.fit()

    y_pred = poly_model.predict(X_poly)
    r2 = r2_score(y_pred, y)

    print(r2, poly_model.theta)

    X_plot = df.Level
    y_func = poly_model.theta[3]*X*X*X + poly_model.theta[2]*X*X + poly_model.theta[1]*X + poly_model.theta[0]
    plt.plot(X_plot, y, 'yo', X, y_func, '--k', color='r')
    plt.show()

    # With sklearn
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X.to_frame())

    poly_model = LinearRegression(X_poly, y, lr=1e-6, num_epochs=500)
    poly_model.fit()

    y_pred = poly_model.predict(X_poly)
    r2 = r2_score(y_pred, y)

    X_plot = df.Level
    y_func = poly_model.theta[2]*X*X + poly_model.theta[1]*X + poly_model.theta[0]
    plt.plot(X_plot, y, 'yo', X, y_func, '--k')
    plt.show()
    
    alpha, beta = optimize.curve_fit(func, xdata = X, ydata = y)[0]
    print(f'alpha={alpha}, beta={beta}')

    y_pred = func(X, alpha, beta)
    r2 = r2_score(y_pred, y)

    plt.plot(X, y, 'b.')
    plt.plot(X, alpha*np.exp(beta*X), 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    main()
