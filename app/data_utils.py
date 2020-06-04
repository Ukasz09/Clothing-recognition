from tensorflow import keras
import matplotlib.pyplot as plt
import math
import csv

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


def load_scaled_and_flatten_data():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images, test_images = scale_data(train_images, test_images)
    train_images = flat_matrix(train_images)
    test_images = flat_matrix(test_images)
    return (train_images, train_labels), (test_images, test_labels)


def split_to_train_and_val(x_data, y_data, percent_of_validation_set=20):  # todo 20 or 5?
    val_batch_size = int(percent_of_validation_set / 100 * len(x_data))
    x_val = x_data[0: val_batch_size]
    y_val = y_data[0: val_batch_size]
    x_train = x_data[val_batch_size:]
    y_train = y_data[val_batch_size:]

    return (x_train, y_train), (x_val, y_val)


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


def split_to_batches(x, batch_size):
    batch_qty = int(len(x) / batch_size)
    batches = [x[i * batch_size:(i + 1) * batch_size] for i in range(batch_qty)]
    return batches


def save_labels_to_csv(labels_list, filename):
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, delimiter=',', quotechar='"')
        for i in range(len(labels_list)):
            wr.writerow([i, labels_list[i]])


def save_result_report(result_params, filename):
    with open(filename, 'w', newline='') as result_file:
        wr = csv.writer(result_file, delimiter=',', quotechar='"')
        wr.writerow(["Name", "Parameter", "Accuracy", "Training time"])
        wr.writerow(result_params)
