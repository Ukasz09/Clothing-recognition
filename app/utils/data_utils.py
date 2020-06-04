from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import math
import csv
import random
import numpy as np

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
COLORS_MAPS = ['inferno', 'viridis']
ROW_SIZE = 28
COL_SIZE = 28
IMG_SHAPE = (ROW_SIZE, COL_SIZE, 1)


# -------------------------------------------------------------------------------------------------------------------- #
def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return (X_train, y_train), (X_test, y_test)


def scale_data(X_train, X_test, value=255.0):
    return X_train.astype('float32') / value, X_test.astype('float32') / value


def scale_data(X_train, X_test, X_val, value=255.0):
    return X_train.astype('float32') / value, X_test.astype('float32') / value, X_val.astype('float32') / value


def change_data_to_3d(X_train, X_test, X_val):
    X_train = X_train.reshape(X_train.shape[0], *IMG_SHAPE)
    X_test = X_test.reshape(X_test.shape[0], *IMG_SHAPE)
    X_val = X_val.reshape(X_val.shape[0], *IMG_SHAPE)
    return X_train, X_test, X_val


def predict_labels(pyx):
    """
    :param pyx: matrix with probability distribution p(y|x) for every class and *X_test* object
    :return: list with predicted class labels
    """
    return [np.argmax(row, axis=0) for row in pyx]


############################3
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


def plot_rand_images(train_images, train_labels, color_map=plt.cm.binary, qty=9, plt_size=10, plt_show=False):
    plt.figure(figsize=(plt_size, plt_size))
    grids_qty = math.ceil(math.sqrt(qty))
    for i in range(qty):
        plt.subplot(grids_qty, grids_qty, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        rand_offset = random.randint(0, train_images.shape[0])
        plt.imshow(train_images[rand_offset], cmap=color_map)
        plt.colorbar()
        plt.xlabel(CLASS_NAMES[train_labels[rand_offset]])
    if plt_show:
        plt.show()


def plot_example_images(all_images, all_labels):
    for cmap in COLORS_MAPS:
        plot_rand_images(all_images, all_labels, color_map=plt.get_cmap(cmap), plt_show=False)
    plt.show()


def plot_image_with_predict_bar(test_images, test_labels, predictions_matrix, predicted_labels, row_num=5, col_num=3):
    """
    Plot the first X test images with predicted and true labels.
    Color predictions:
    - green: correct
    - red: incorrect

    :param predictions_matrix: matrix with probability distribution p(y|x) for every class and *x_test* object
    """
    num_images = row_num * col_num
    plt.figure(figsize=(2 * 2 * col_num, 2 * row_num))
    for i in range(num_images):
        plt.subplot(row_num, 2 * col_num, 2 * i + 1)
        __plot_image_with_axis(i, predictions_matrix[i], predicted_labels[i], test_labels, test_images)
        plt.subplot(row_num, 2 * col_num, 2 * i + 2)
        __plot_predict_arr_graph(i, predictions_matrix[i], predicted_labels[i], test_labels)
    plt.tight_layout()
    plt.show()


def __plot_image_with_axis(i, predictions_array, predicted_label, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label], 100 * np.max(predictions_array),
                                         CLASS_NAMES[true_label]), color=color)


def __plot_predict_arr_graph(i, predictions_array, predicted_label, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')