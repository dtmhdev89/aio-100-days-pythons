import numpy as np
import random

import matplotlib.pyplot as plt

from dataset_man import dataset_manager

def get_column(data, index):
    return [ row[index] for row in data ]

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    N = len(data)

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    X = [tv_data, radio_data, newspaper_data]
    y = sales_data

    return X, y

def prepare_data_v2(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()
    N = len(data)

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    X = [ [1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data) ]
    y = sales_data

    return X, y

def initialize_params():
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    # b = 0

    w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)

    return w1, w2, w3, b

def initialize_params_v2():
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    # b = 0

    w1, w2, w3, b = (-0.01268850433497871, 0.004752496982185252, 0.0073796171538643845, 0)

    return [b, w1, w2, w3]

def predict(x1, x2, x3, w1, w2, w3, b):
    return x1*w1 + x2*w2 + x3*w3 + b

def predict_v2(X_features, weights):
    return sum([xi*b for xi, b in zip(X_features, weights)])

def compute_loss(y, y_hat, metric='mse'):
    if metric == 'mse':
        return compute_loss_mse(y, y_hat)
    elif metric == 'mae':
        return compute_loss_mae(y, y_hat)
    
def compute_loss_mae(y, y_hat):
    return abs(y_hat - y)

def compute_loss_mse(y, y_hat):
    return (y_hat - y)**2

def compute_gradient_w(X_features, y, y_hat):
    dl_dweights = [2*xi*(y_hat-y) for xi in x_i]
    
    return dl_dweights

def update_weight(weights, dl_weights, lr):
    return [(w - lr * dl_w) for w, dl_w in zip(weights, dl_weights)]

def compute_gradient_wi(x_i, y, y_hat):
    dl_wi = 2*x_i*(y_hat - y)

    return dl_wi

def compute_gradient_b(y, y_hat):
    dl_db = 2*(y_hat - y)

    return dl_db

def update_weight_wi(w_i, dl_dwi, lr):
    return w_i - lr*dl_dwi

def update_weight_b(b, dl_db, lr):
    return b - lr*dl_db

def implement_linear_regression(X_data, y_data, epoch_max=50, lr=1e-5, metric='mse'):
    losses = []
    tv_idx = 0
    radio_idx = 1
    newspaper_idx = 2

    w1, w2, w3, b = initialize_params()

    N = len(y_data)

    for epoch in range(epoch_max):
        for i in range(N):
            x1 = X_data[tv_idx][i]
            x2 = X_data[radio_idx][i]
            x3 = X_data[newspaper_idx][i]

            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            
            loss = compute_loss(y, y_hat, metric)

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            losses.append(loss)
    
    return (w1, w2, w3, b, losses)

def implement_linear_regression_v2(X_feature, y_output, epoch_max=50, lr=1e-5, metric='mse'):
    losses = []

    weights = initialize_params_v2()

    N = len(y_output)

    for epoch in range(epoch_max):
        for i in range(N):
            feature_i = X_feature[i]

            y = y_output[i]

            y_hat = predict_v2(feature_i, weights)
            
            loss = compute_loss(y, y_hat, metric)

            dl_dweights = compute_gradient_w(feature_i, y, y_hat)

            weights = update_weight(weights, dl_dweights, lr)

            losses.append(loss)
    
    return weights, losses

def implement_linear_regression_nsamples(X_data, y_data, epoch_max=50, lr=1e-5, metric='mse'):
    losses = []
    tv_idx = 0
    radio_idx = 1
    newspaper_idx = 2

    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for epoch in range(epoch_max):
        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0

        for i in range(N):
            x1 = X_data[tv_idx][i]
            x2 = X_data[radio_idx][i]
            x3 = X_data[newspaper_idx][i]

            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            
            loss = compute_loss(y, y_hat, metric)
            loss_total += loss

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db
        
        w1 = update_weight_wi(w1, dw1_total / N, lr)
        w2 = update_weight_wi(w2, dw2_total / N, lr)
        w3 = update_weight_wi(w3, dw3_total / N, lr)
        b = update_weight_b(b, db_total / N, lr)

        losses.append(loss_total / N)

    return (w1, w2, w3, b, losses)

def main():
    advertising_dataset_path = dataset_manager.get_dataset_path_by_id('m4.advertising')
    X, y = prepare_data(advertising_dataset_path)

    print('---Cau 1:')
    sum_list = [sum(X[0][:5]), sum(X[1][:5]),sum(X[2][:5]), sum(y[:5])]
    print(sum_list)

    print('---Cau 2:')
    y_2 = predict(x1=1, x2=1 , x3=1, w1=0, w2=0.5, w3=0, b=0.5)
    print(y_2)

    print('---Cau 3:')
    l = compute_loss_mse(y_hat=1, y=0.5)
    print(l)

    print('---Cau 4:')
    g_wi = compute_gradient_wi(x_i=1.0 , y=1.0, y_hat=0.5)
    print(g_wi)

    print('---Cau 5:')
    g_b = compute_gradient_b(y=2.0, y_hat=0.5)
    print(g_b)

    print('---Cau 6:')
    after_wi = update_weight_wi(w_i=1.0, dl_dwi=-0.5, lr=1e-5)
    print(after_wi)

    print('---Cau 7:')
    after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
    print(after_b)

    w1, w2, w3, b, losses = implement_linear_regression(X, y)
    plt.plot(losses[:100])
    plt.xlabel('#iteration')
    plt.ylabel('Loss')
    plt.show()

    print('---Cau 8:')
    print(w1, w2, w3)

    print('---Cau 9:')
    tv = 19.2
    radio = 35.9
    newspaper = 51.3

    X , y = prepare_data(advertising_dataset_path)
    (w1, w2, w3, b, losses) = implement_linear_regression(X, y, epoch_max=50, lr=1e-5)
    sales = predict(tv, radio, newspaper, w1, w2, w3, b)
    print (f'predicted sales is {sales}')

    print('---Cau 10:')
    X, y = prepare_data(advertising_dataset_path)
    (w1, w2, w3, b, losses) = implement_linear_regression(X, y, epoch_max=50, lr=1e-5, metric='mae')
    plt.plot(losses[:100])
    plt.xlabel('#iteration')
    plt.ylabel('Loss')
    plt.show()

    print('---Cau 11:')
    X, y = prepare_data(advertising_dataset_path)
    (w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr=1e-5, metric='mse')
    print(w1, w2, w3)

    plt.plot(losses)
    plt.xlabel('#epoch')
    plt.ylabel('MSE Loss')
    plt.show()

    (w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, epoch_max=1000, lr=1e-5, metric='mae')
    print(w1, w2, w3)

    plt.plot(losses)
    plt.xlabel('#epoch')
    plt.ylabel('MAE Loss')
    plt.show()

    print('---Cau 12:')
    X, y = prepare_data_v2(advertising_dataset_path)
    W, L = implement_linear_regression_v2(X, y, epoch_max=1000, lr=1e-5, metric='mse')
    print(W)
    print(L[9999])

if __name__ == "__main__":
    main()
