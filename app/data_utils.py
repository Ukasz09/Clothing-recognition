from tensorflow import keras
import matplotlib.pyplot as plt
import math

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def scale_data(train_images, test_images, value=255.0):
    return train_images / value, test_images / value


def flat_matrix(matrix):
    x, y, z = matrix.shape[0], matrix.shape[1], matrix.shape[2]
    matrix = matrix.reshape(x, y * z)
    return matrix


def display_first_images(train_images, train_labels, color_map=plt.cm.binary, qty=25, plt_size=10):
    plt.figure(figsize=(plt_size, plt_size))
    grids_qty = math.ceil(math.sqrt(qty))
    for i in range(qty):
        plt.subplot(grids_qty, grids_qty, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=color_map)
        plt.colorbar()
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


def split_to_batches(x, batch_size=2500):  # todo 15 000
    batch_qty = int(len(x) / batch_size)
    batches = [x[i * batch_size:(i + 1) * batch_size] for i in range(batch_qty)]
    return batches
