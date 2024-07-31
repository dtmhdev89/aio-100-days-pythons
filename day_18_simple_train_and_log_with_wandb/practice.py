import pandas as pd
# import wandb

# forward
def predict(x, w, b):
    return x * w + b

def gradient(y_hat, y, x):
    dw = 2 * x * (y_hat - y)
    db = 2 * (y_hat - y)

    return (dw, db)

def compute_loss(y_hat, y):
    return (y_hat - y)**2 / 2.0

# update weights
def update_weights(w, b, lr, dw, db):
    w_new = w - lr * dw
    b_new = b - lr * db

    return (w_new, b_new)

def main():
    areas = [6.7, 4.6, 3.5, 5.5]
    prices = [9.1, 5.9, 4.6, 6.7]

    dataset = pd.DataFrame({
        'areas': areas,
        'prices': prices
    })

    b = 0.04
    w = -0.34
    lr = 0.01
    epochs = 10

    # wandb.init(
    #     project="demo-linear-regression",
    #     config = {
    #         "learning_rate": lr,
    #         "epochs": epochs
    #     }
    # )

    # wandb.run.log({"Dataset": wandb.Table(dataframe=dataset)})

    X_train = dataset['area']
    Y_train = dataset['price']

    N = len(X_train)

    losses = []

    for epoch in range(epochs):
        for in range(N):
            x = X_train[i]
            y = Y_train[i]

            y_hat = predict(x, w, b)
            loss = compute_loss(y_hat, y)

            wandb.log({"loss": loss})

            (dw, db) = gradient(y_hat, y, x)

            (w, b) = update_weights(w, b, lr, dw, db)

    # wandb.finish()

if __name__ == "__main__":
    main()
