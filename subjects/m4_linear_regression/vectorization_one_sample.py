import numpy as np

import matplotlib.pyplot as plt

def predict(X, theta):
    return X.dot(theta)

def compute_loss(y, y_hat):
    return (y_hat - y)**2

def compute_gradient(X, y, y_hat):
    return 2*X*(y_hat - y)

def update_weights(theta, dl_dtheta, lr):
    return theta - lr * dl_dtheta

def main():
    # Experience và Education
    X = np.array([[3, 12], [4, 13], [5, 14], [6, 15]])

    X_bias = np.c_[np.ones(X.shape[0]), X]
    # alternative for np.c_
    # X_bias = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)
    print(X_bias)

    # Salary thực tế
    y = np.array([60, 55, 66, 93])

    # [b, w1, w2] - khởi tạo tham số
    theta = np.array([10, 3, 2])

    print(f'Input shape: {X_bias.shape}')
    print(f'Output shape: {y.shape}')
    print(f'Theta shape: {theta.shape}')

    N = len(y)
    n_epochs = 4
    lr = 1e-3
    losses = []
    for epoch in range(n_epochs):
        epoch_losses = []

        for i in range(N):
            X_i = X_bias[i, :]
            y_i = y[i]
            y_hat = predict(X_i, theta)
            
            loss = compute_loss(y_i, y_hat)
            epoch_losses.append(loss)

            dl_dtheta = compute_gradient(X_i, y_i, y_hat)

            theta = update_weights(theta, dl_dtheta, lr)
    
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(epoch_loss)
        print(f'Epoch {epoch+1}: Loss = {epoch_loss}, theta = {theta}')

    # After training prediction
    X_test = np.array([[1, 4, 13]])
    y_hat = predict(X_test, theta)
    loss = compute_loss(y_hat, y)
    print(f'Loss after training: {loss}')

    plt.plot(losses)
    plt.title('Loss over epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
