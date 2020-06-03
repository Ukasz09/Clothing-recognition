import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import math

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    return (train_images, train_labels), (test_images, test_labels)


def scale_data(train_images, test_images, value=255.0):
    return train_images / value, test_images / value


def display_first_images(train_images, train_labels, qty=25):
    plt_size = 0.4 * qty
    plt.figure(figsize=(plt_size, plt_size))
    grids_qty = math.ceil(math.sqrt(qty))
    for i in range(qty):
        plt.subplot(grids_qty, grids_qty, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()
