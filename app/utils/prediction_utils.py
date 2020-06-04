import time
import datetime
import numpy as np

__start_time = time.time()
__end_time = time.time()


def calc_accuracy(predicted_labels, real_labels):
    correct_qty = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == real_labels[i]:
            correct_qty += 1
    return correct_qty * 100 / len(predicted_labels)


def predict_labels(pyx):
    """
    :param pyx: matrix with probability distribution p(y|x) for every class and *X_test* object
    :return: list with predicted class labels
    """
    return [np.argmax(row, axis=0) for row in pyx]


def convert_time(sec):
    return str(datetime.timedelta(seconds=sec))
