import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
matplotlib.use('QtAgg')

from IPython.display import HTML
matplotlib.rcParams.update({'font.size': 14})
plt.style.use('seaborn-v0_8-white')


def f(x):
    return x*x


def compute_loss(x):
    return x**2


x_min = -100
x_max = 100

x = np.linspace(x_min, x_max, 300)
y = f(x)

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim([x_min, x_max])
ax.set_ylim([-100, 10000])
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
line, = ax.plot(x, y)
plt.show()

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
print(opt)

list_grad = []
list_x = []

var = tf.Variable(70.0)
list_x.append(var.numpy())
print('var: ', var.numpy())

lr = 0.1
for i in range(100):
    with tf.GradientTape() as tape:
        loss = compute_loss(var)

    grads = tape.gradient(loss, var)
    list_grad.append(grads.numpy())

    var.assign_add(-lr*grads)  
    list_x.append(var.numpy())

plt.plot(list_grad[80:], label='grads')
plt.plot(list_x[80:100], label='x')
plt.xlabel('iteration')
plt.ylabel('Value')
plt.legend()
plt.show()
