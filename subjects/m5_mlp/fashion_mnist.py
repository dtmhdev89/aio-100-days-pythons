import numpy as np
import gzip
import os
import matplotlib.pyplot as plt
from PIL import Image
from urllib import request


def build_file_path(data_path, file_name):
    return os.path.join(data_path, file_name)

if __name__ == "__main__":
    download_dataset = False
    if download_dataset:
        filenames = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        ]

        folder = 'data_fashion_mnist/'
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        for name in filenames:
            print("Downloading " + name + "...")

            # lưu vào folder data_fashion_mnist
            request.urlretrieve(base_url + name, folder + name)

    data_path = os.path.join('data')
    img_flatten_shape = 28 * 28
    train_img_path = build_file_path(data_path, 'train-images-idx3-ubyte.gz')
    with gzip.open(train_img_path, 'rb') as f:
        X_train = np.frombuffer(f.read(),
                                np.uint8,
                                offset=16).reshape(-1, img_flatten_shape)

    test_img_path = build_file_path(data_path, 't10k-images-idx3-ubyte.gz')
    with gzip.open(test_img_path, 'rb') as f:
        X_test = np.frombuffer(f.read(),
                               np.uint8,
                               offset=16).reshape(-1, img_flatten_shape)

    y_train_path = build_file_path(data_path, 'train-labels-idx1-ubyte.gz')
    with gzip.open(y_train_path, 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)

    y_test_path = build_file_path(data_path, 't10k-labels-idx1-ubyte.gz')
    with gzip.open(y_test_path, 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    indices = list(np.random.randint(X_train.shape[0], size=144))
    print(len(indices))

    fig = plt.figure(figsize=(9, 9))
    columns, rows = 12, 12

    for i in range(1, columns * rows + 1):
        img = X_train[indices[i-1]].reshape(28, 28)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    # plt.show()

    # pil_indices = list(np.random.randint(6000, size=10))
    # os.makedirs(os.path.join('images'), exist_ok=True)
    
    # for i in range(10):
    #     im = Image.fromarray(X_train[pil_indices[i]].reshape(28, 28))
    #     im.save(os.path.join("images", f"image_{str(i)}.png"))
