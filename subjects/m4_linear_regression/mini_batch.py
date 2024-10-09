import numpy as np
import matplotlib.pyplot as plt

def predict(w, b, x):
    return w*x +b

def compute_loss(y_hat, y):
    return (y_hat - y)**2

def compute_gradient(y_hat, y, x):
    dl_dw = 2*x*(y_hat - y)
    dl_db = 2*(y_hat - y)
    
    return dl_dw, dl_db

def update(w, b, lr, dl_dw, dl_db):
    w = w - lr*dl_dw
    b = b - lr*dl_db

    return w, b

def main():
    # initial
    areas = [6.7, 4.6, 3.5, 5.5]
    prices = [9.1, 5.9, 4.6, 6.7]
    w = -0.34
    b = 0.04
    lr = 0.01

    # plot initial
    plt.scatter(areas, prices)
    x_plot = np.arange(3, 7, 0.1)
    y_initial = [predict(w, b, x_i) for x_i in x_plot]
    plt.plot(x_plot, y_initial, 'r')

    # plt.xlim(0, 7)
    # plt.ylim(-3, 10)
    plt.show()

    # using single sample
    losses = []
    for i in range(len(areas)):
        price_pred = predict(w, b, areas[i])

        # compute loss for comparing (optional step in algorithm)
        loss = compute_loss(price_pred, prices[i])
        losses.append(loss)

        # calculate gradient
        dl_dw, dl_db = compute_gradient(price_pred, prices[i], areas[i])

        # update parameters
        w, b = update(w, b, lr, dl_dw, dl_db)

    # plot losses to see if we were in right way
    plt.plot(losses)
    plt.show()

    # plot updated w, b to see if the linear is good
    print(f'latest w ({w}), b ({b})')
    plt.scatter(areas, prices)
    x_plot = np.arange(3, 7, 0.1)
    y_after_update = [predict(w, b, x_i) for x_i in x_plot]
    plt.plot(x_plot, y_after_update, 'r')
    plt.show()

    w = -0.34
    b = 0.04
    lr = 0.01

    N = len(areas)
    m = 2
    epochs = 32

    # using mini-batch and epochs training
    losses = []
    for _ in range(epochs):
        for i in range(0, N, m):
            acc_dl_dw = 0
            acc_dl_db = 0
            acc_loss = 0
            
            for j in range(m):
                y_hat = predict(w, b, areas[i + j])

                loss = compute_loss(y_hat, prices[i + j])
                acc_loss += loss

                dl_dw, dl_db = compute_gradient(y_hat, prices[i + j], areas[i + j])
                # print(f'y_hat ({i}): {y_hat};')
                # print(f'dl_dw ({i}): {dl_dw};\tdl_db ({i}): {dl_db}')
                acc_dl_dw += dl_dw
                acc_dl_db += dl_db
            
            # update w of all m samples' gradient
            losses.append(acc_loss / m)
            avg_dl_dw = acc_dl_dw / m
            avg_dl_db = acc_dl_db / m
            w, b = update(w, b, lr, avg_dl_dw, avg_dl_db)
            # print(f'w ({w}), b ({b})')
    
    # w, b after epochs trained
    print(f'w ({w}); b ({b})')

    # plot losses
    _fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(losses)
    axes[1].plot(losses[-20:])
    plt.show()

    # plot predicted line
    plt.scatter(areas, prices)
    x_plot = np.arange(3, 7, 0.1)
    y_mini_batch_after_update = [predict(w, b, x_i) for x_i in x_plot]
    plt.plot(x_plot, y_mini_batch_after_update, 'r')
    plt.show()
        

if __name__ == "__main__":
    main()
