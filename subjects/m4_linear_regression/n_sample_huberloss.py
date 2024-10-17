import numpy as np
import matplotlib.pyplot as plt

from prepare_dataset import DataDataset

def predict(x,w,b):
    return x*w + b

def compute_loss(y_hat, y, delta=5):
    if abs(y_hat-y) < delta:
        loss = (y_hat - y)*(y_hat - y)
    else:
        loss = delta*abs(y_hat-y) - 0.5*delta*delta
    return loss

def gradient_sr(y_hat, y, x):
    dw = 2*x*(y_hat-y)
    db = 2*(y_hat-y)
    
    return (dw, db)

def gradient_ab(y_hat, y, x, delta):
    dw = delta*x*(y_hat-y)/abs(y_hat-y)
    db = delta*(y_hat-y)/abs(y_hat-y)
    
    return (dw, db)

def gradient(y_hat, y, x, delta=5):
    if abs(y_hat-y) < delta:
        dw, db = gradient_sr(y_hat, y, x)
    else:
        dw, db = gradient_ab(y_hat, y, x, delta)

    return (dw, db)

def update_weight(w, b, lr, dw, db):
    w_new = w - lr*dw
    b_new = b - lr*db
    
    return (w_new, b_new)

def main():
    data = DataDataset()
    X, y = data.X, data.y

    b = 0.04
    w = -0.34
    lr = 0.01

    epoch_max = 30
    N = len(y)

    losses = [] # for debug
    for epoch in range(epoch_max):
        
        (dw_total, db_total) = (0, 0)
        loss_total = 0.0
        for i in range(N):
            xi = X[i]
            yi = y[i]
            
            y_hat = predict(xi, w, b)

            loss = compute_loss(y_hat, yi)
            loss_total = loss_total + loss 

            (dw, db) = gradient(y_hat, yi, xi)

            dw_total = dw_total + dw
            db_total = db_total + db
            
        losses.append(loss_total/N)
            
        (w, b) = update_weight(w, b, lr, dw_total/N, db_total/N)
            
    print(w, b)

    x_data = range(2, 8)
    y_data = [x*w + b for x in x_data]
    plt.plot(x_data, y_data, 'r')

    areas  = X
    prices = y
    plt.scatter(areas, prices)

    plt.xlabel('Area (x 100$m^2$)')
    plt.ylabel('Price (Tael)')
    plt.title('Huber Loss (Batch)')
    plt.show()

if __name__ == "__main__":
    main()
