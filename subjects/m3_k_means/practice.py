import numpy as np

X = np.array([
    [2.0, 3.0, 1.5],
    [3.0, 3.5, 2.0],
    [3.5, 3.0, 2.5],
    [8.0, 8.0, 7.5],
    [8.5, 8.5, 8.0],
    [9.0, 8.0, 8.5],
    [1.0, 2.0, 1.0],
    [1.5, 2.5, 1.5]
])

C = np.array([
    [2.0, 3.0, 1.5],
    [1.0, 2.0, 1.0]
])

np.random.seed(42)
print(X[np.random.choice(X.shape[0], 2, replace=False)])

# X_reshape = X.reshape(8, 1, 3)

# def l2_norm(v_X, v_C):
#     return np.sqrt(np.sum((v_X - v_C)**2, axis=-1))

# print(X.shape)
# print(C.shape)

# print(l2_norm(X_reshape, C))

# print(np.subtract(X - C))
