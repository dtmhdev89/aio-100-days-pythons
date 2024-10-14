import numpy as np
import matplotlib.pyplot as plt
import sys

from prepare_data import AdvertisingDataset

def mean_normalization(X):
    N = len(X)
    maxi = np.max(X)
    mini = np.min(X)
    avg = np.mean(X)
    X = (X - avg) / (maxi - mini)
    X_b = np.c_[np.ones((N, 1)), X]

    return X_b, maxi, mini, avg

def sto_predict(X, w):
    return X.dot(w)

def compute_sto_loss(y_hat, y):
    return ((y_hat - y) ** 2) / 2

def sto_compute_gradient(X, y_hat, y):
    return X.T.dot(y_hat - y)

def sto_update_weights(thetas, dl_dthetas, lr):
    return thetas - lr * dl_dthetas

def stochastic_gradient_descent(X_b, y, n_epochs=50, lr=1e-5):
    thetas = np.asarray([1.16270837, -0.81960489, 1.39501033, 0.29763545]).reshape(-1, 1)

    thetas_path = [thetas]
    losses = []

    N = X_b.shape[0]

    for epoch in range(n_epochs):
        for i in range(N):
            # select random number in N
            # random_index = np.random.randint(N) # should use this in real
            random_index = i
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]

            y_hat = sto_predict(xi, thetas)

            loss = compute_sto_loss(y_hat, yi)

            dl_dthetas = sto_compute_gradient(xi, y_hat, yi)

            thetas = sto_update_weights(thetas, dl_dthetas, lr)

            losses.append(loss.flatten())
            thetas_path.append(thetas)
    
    return thetas_path, losses

def mini_predict(X, w):
    return X.dot(w)

def compute_mini_loss(y_hat, y):
    return ((y_hat - y)**2) / 2

def mini_compute_gradient(X, y_hat, y):
    return X.T.dot(y_hat - y)

def mini_update_weights(thetas, dl_dthetas, lr):
    return thetas - lr * dl_dthetas

def mini_batch_gradient_descent(X_b, y, n_epochs=50, minibatch_size=20, lr=1e-2):
    #thetas = np.random.randn(4, 1)
    thetas = np.asarray([1.16270837, -0.81960489, 1.39501033, 0.29763545]).reshape(-1, 1)
    thetas_path = [thetas]
    losses = []
    N = X_b.shape[0]

    for epoch in range(n_epochs):
        # shuffled_indices = np.random.permutation(N) # for real

        shuffled_indices = np.asanyarray([21, 144, 17, 107, 37, 115, 167, 31, 3,
            132, 179, 155, 36, 191, 182, 170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16,
            185, 11, 1, 34, 177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126,
            165, 78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190,
            169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131,
            77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139,
            195, 72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147,
            92, 52, 124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47,
            174, 100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67,
            129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24,
            168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55,
            133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122,
            154])
        
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        for i in range(0, N, minibatch_size):
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size].reshape(-1, 1)

            y_hat = mini_predict(xi, thetas)

            loss = compute_mini_loss(y_hat, yi)

            dl_dthetas = mini_compute_gradient(xi, y_hat, yi)

            thetas = mini_update_weights(thetas, dl_dthetas/minibatch_size, lr)

            thetas_path.append(thetas)
            loss_mean = np.sum(loss) / minibatch_size
            losses.append(loss_mean)
    
    return thetas_path, losses

def batch_predict(X, w):
    return X.dot(w)

def batch_loss(y_hat, y):
    return (y_hat - y) ** 2

def compute_batch_loss_gradient(y_hat, y, N):
    return 2 * (y_hat - y) / N

def compute_batch_gradient(X, dl_dyhat):
    return X.T.dot(dl_dyhat)

def update_batch_weights(thetas, dl_dthetas, lr):
    return thetas - lr * dl_dthetas

def batch_gradient_descent(X_b, y, n_epochs=100, lr=1e-2):
    thetas = np.asarray([1.16270837, -0.81960489, 1.39501033, 0.29763545]).reshape(-1, 1)
    thetas_path = [thetas]
    losses = []
    N = X_b.shape[0]

    for epoch in range(n_epochs):
        y_hat = batch_predict(X_b, thetas)

        loss = batch_loss(y_hat, y)

        dl_dyhat = compute_batch_loss_gradient(y_hat, y, N)

        dl_dthetas = compute_batch_gradient(X_b, dl_dyhat)

        thetas = update_batch_weights(thetas, dl_dthetas, lr)

        thetas_path.append(thetas)
        mean_loss = np.sum(loss) / N
        losses.append(mean_loss)

    return thetas_path, losses

def main():
    advertising_data = AdvertisingDataset()
    X = advertising_data.X
    y = advertising_data.y

    X_b, maxi, mini, avg = mean_normalization(X)

    # sgd_theta, losses = stochastic_gradient_descent(X_b, y, n_epochs=50, lr=1e-2)
    # x_axis = list(range(500))
    # plt.plot(x_axis, losses[:500], color='r')
    # plt.show()

    sgd_theta, losses = stochastic_gradient_descent(X_b, y, n_epochs=1, lr=1e-2)
    print(np.sum(losses))

    mbgd_thetas, losses = mini_batch_gradient_descent(X_b, y, n_epochs=50, minibatch_size=20, lr=1e-2)
    print(round(sum(losses), 2))
    # x_axis = list(range(200))
    # plt.plot(x_axis, losses[:200], color='r')
    # plt.show()

    bgd_thetas, losses = batch_gradient_descent(X_b, y.reshape(-1, 1), n_epochs=100, lr=1e-2)
    print(round(sum(losses), 2))
    # x_axis = list(range(100))
    # plt.plot(x_axis, losses[:100], color='r')
    # plt.show()

if __name__ == "__main__":
    main()
