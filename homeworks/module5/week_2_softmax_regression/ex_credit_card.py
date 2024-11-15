import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dataset_man import dataset_manager

def compute_softmax(z):
    exp_z = np.exp(z)

    return exp_z / exp_z.sum(axis=1)[:, None]

def predict(X, theta):
    z = np.dot(X, theta)
    y_hat = compute_softmax(z)

    return y_hat

def compute_loss(y_hat, y):
    n = y.size

    return (-1/n) * np.sum(y * np.log(y_hat))

def compute_gradient(X, y, y_hat):
    n = y.size

    return np.dot(X.T, (y_hat - y)) / n

def update_theta(theta, gradient, lr):
    return theta - lr * gradient

def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta)
    acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()

    return acc

def main():
    df = pd.DataFrame(dataset_manager.load_gdrive_dataset('11EBwEw6yNRWgNzi7XiLXABpfYG_q7iHY', as_dataframe=True))
    print(df.info())
    # print(df.isna().sum())
    # print(df.head())

    dataset_arr = df.to_numpy()

    X = dataset_arr[:, :-1].astype(np.float64)
    y = dataset_arr[:, -1].astype(np.uint8)
    print("Labels: ", np.unique(y, return_counts=True))

    intercept = np.ones((X.shape[0], 1))
    X_b = np.concatenate((intercept, X), axis=1)

    n_classes = np.unique(y, axis=0).shape[0]
    n_samples = y.shape[0]

    print(f"n classes: {n_classes}")
    print(f"n sampes: {n_samples}")

    y_encoded = np.array(
        [np.zeros(n_classes) for _ in range(n_samples)]
    )
    y_encoded[np.arange(n_samples), y] = 1

    val_size = 0.2
    test_size = 0.125
    random_state = 2
    is_shuffle = True

    X_train, X_val, y_train, y_val = train_test_split(
        X_b, y_encoded,
        test_size=val_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    normalizer = StandardScaler()
    X_train[:, 1:] = normalizer.fit_transform(X_train[:, 1:])
    X_val[:, 1:] = normalizer.transform(X_val[:, 1:])
    X_test[:, 1:] = normalizer.transform(X_test[:, 1:])

    lr = 1e-2
    epochs = 30
    batch_size = 1024
    n_features = X_train.shape[1]

    np.random.seed(random_state)
    theta = np.random.uniform(size=(n_features, n_classes))

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in range(epochs):
        train_batch_losses = []
        train_batch_accs = []
        val_batch_losses = []
        val_batch_accs = []

        for i in range(0, X_train.shape[0], batch_size):
            X_i = X_train[i:(i + batch_size)]
            y_i = y_train[i:(i + batch_size)]

            y_hat = predict(X_i, theta)
            
            train_loss = compute_loss(y_hat, y_i)
            gradient = compute_gradient(X_i, y_i, y_hat)
            theta = update_theta(theta, gradient, lr)

            train_batch_losses.append(train_loss)
            train_acc = compute_accuracy(X_train, y_train, theta)
            train_batch_accs.append(train_acc)

            y_val_hat = predict(X_val, theta)
            val_loss = compute_loss(y_val_hat, y_val)
            val_batch_losses.append(val_loss)

            val_acc = compute_accuracy(X_val, y_val, theta)
            val_batch_accs.append(val_acc)

        train_batch_loss = sum(train_batch_losses) / len(train_batch_losses)
        val_batch_loss = sum(val_batch_losses) / len(val_batch_losses)
        train_batch_acc = sum(train_batch_accs) / len(train_batch_accs)
        val_batch_acc = sum(val_batch_accs) / len(val_batch_accs)

        train_losses.append(train_batch_loss)
        val_losses.append(val_batch_loss)
        train_accs.append(train_batch_acc)
        val_accs.append(val_batch_acc)

        print (f'\nEPOCH {epoch + 1}:\tTraining loss: {train_batch_loss:.3f}\tValidation loss: {val_batch_loss:.3f}')

    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Top left plot: Training Loss
    ax[0, 0].plot(train_losses)
    ax[0, 0].set(xlabel='Epoch', ylabel='Loss', title='Training Loss')

    # Top right plot: Validation Loss
    ax[0, 1].plot(val_losses, 'orange')
    ax[0, 1].set(xlabel='Epoch', ylabel='Loss', title='Validation Loss')

    # Bottom left plot: Training Accuracy
    ax[1, 0].plot(train_accs)
    ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy', title='Training Accuracy')

    # Bottom right plot: Validation Accuracy
    ax[1, 1].plot(val_accs, 'orange')
    ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy', title='Validation Accuracy')

    plt.tight_layout()
    plt.show()

    val_set_accuracy = compute_accuracy(X_val, y_val, theta)
    test_set_accuracy = compute_accuracy(X_test, y_test, theta)

    print("Evaluation on validation and test sets:")
    print(f"Validation Accuracy: {val_set_accuracy}")
    print(f"Test Accuracy: {test_set_accuracy}")

if __name__ == "__main__":
    main()
