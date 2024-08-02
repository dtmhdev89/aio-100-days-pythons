import numpy as np


"""
y = A*x
AT * A * x = AT * y
x = (AT * A)-1 * AT * y

(AT * A) is symmetric and invertible
"""
def compute_loss(x, y, area_np):
    y_hat = x[0] * area_np + x[1]
    loss = np.divide(np.sum(np.abs(y - y_hat)), len(y_hat))
    print(area_np)
    print(y)
    print(y_hat)
    print(loss)

def main():
    feature_area = np.array([6.7, 4.6, 3.5, 5.5])
    label_price = np.array([9.1, 5.9, 4.6, 6.7])
    # find linear that fits price = a * area + b
    # np array that has Arr shape (n,) that could be treated as Arr.T
    y = np.array(label_price)
    A = np.vstack([feature_area, np.ones(len(feature_area))]).T
    AT = A.T
    ATxA = np.dot(AT, A)
    det_ATxA = np.linalg.det(ATxA)
    
    if det_ATxA != 0:
        ATxA_1 = np.linalg.inv(ATxA)
        x = np.dot(ATxA_1, np.dot(AT, y))
        print(f"price = {round(x[0], 4)} * x + {round(x[1], 4)}")

        compute_loss(x, y, np.array(feature_area))
    else:
        print('ATxA is not invertible')
    

if __name__ == "__main__":
    main()
