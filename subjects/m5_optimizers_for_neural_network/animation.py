from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
matplotlib.use('QtAgg')
matplotlib.rcParams.update({'font.size': 14})

plt.style.use('seaborn-v0_8-white')


def f(x):
    return x ** 2


def compute_loss(var):
    return var ** 2


def init_visualization(fig, ax, x_min, x_max, y_min, y_max):
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    plt.title("SGD")
    line, = ax.plot([], [])
    scat = ax.scatter([], [], c="red")
    text = ax.text(-10, 6000, "", c="green")

    return line, scat, text


def init_animation(line):
    line.set_data([], [])
    return line,


# Animation update function
def animate(frame, var, optimizer, line, scat, text, x, y):
    """Update the plot dynamically."""
    # Perform a single optimization step
    with tf.GradientTape() as tape:
        loss = compute_loss(var)
    grads = tape.gradient(loss, [var])
    optimizer.apply_gradients(zip(grads, [var]))
    x_est = var.numpy()
    y_est = f(x_est)
    grad = 2 * x_est

    # step_count = grads[0].numpy()

    # Update plot elements
    scat.set_offsets([[x_est, y_est]])
    text.set_text(
        f"Step = {frame + 1} - \n"
        f"x = {x_est:.3f}\n"
        f"f(x) = {y_est:.3f}\n"
        f"f'(x) = {grad:.3f}"
    )
    line.set_data(x, y)

    return line, scat, text


learning_rate = 0.1
initial_value = 99.0
x_min, x_max = -100, 100
y_min, y_max = -100, 10000
frames = 3

optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
var = tf.Variable(initial_value)

x = np.linspace(x_min, x_max, 300)
y = f(x)

fig, ax = plt.subplots(figsize=(12, 9))
line, scat, text = init_visualization(fig, ax, x_min, x_max, y_min, y_max)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=30,
    init_func=lambda: init_animation(line),
    fargs=(var, optimizer, line, scat, text, x, y),
    interval=600,
    blit=True
)

writervideo = animation.FFMpegWriter(fps=2, bitrate=1200)
ani.save('1.square_function.mp4', writer=writervideo)

# HTML(ani.to_html5_video())

plt.close(fig)
