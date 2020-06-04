import numpy as np
import time
from app.data_utils import split_to_batches, split_to_train_and_val
from app.prediction_utils import get_time_result

DISTANCE_CALC_METHOD = "euclidean distance (L2)"


def candidate_k_values(min_k=1, max_k=25, step=1):
    """
    :return: list of candidates k values to check
    """
    return range(min_k, max_k, step)


def predict_prob(x_test, x_train, y_train, k):
    """
    :param x_test: test data matrix [N1xW]
    :param x_train: training data matrix [N2xW]
    :param y_train: real class labels for *x_train* object [N2X1]
    :param k: amount of nearest neighbours
    :return: matrix with probability distribution p(y|x) for every class and *x_test* object [N1xM]
    """
    distances = euclidean_distance(x_test, x_train)
    sorted_labels = sort_train_labels(distances, y_train)
    return p_y_x(sorted_labels, k)


def predict_prob_with_splitting_to_batches(x_test, x_train, y_train, k, batch_size):
    """
    Split *x_test* to batches and for each one calc matrix with probability distribution p(y|x) for every y class
    :param k: amount of nearest neighbours
    :return: list of matrices with probability distribution p(y|x) for every x_test batch
    """
    if batch_size < len(x_test):
        test_batches = split_to_batches(x_test, batch_size)
        batches_qty = len(test_batches)
        y_prob = [predict_prob(test_batches[i], x_train, y_train, k) for i in range(batches_qty)]
        return y_prob
    return [predict_prob(x_test, x_train, y_train, k)]


def predict_labels(pyx_knn):
    """
    :param pyx_knn: matrix with probability distribution p(y|x) for every class and *x_test* object  [N1xM]
    :return: list with predicted class labels
    """
    return [np.argmax(row, axis=0) for row in pyx_knn]


def predict_labels_for_every_batch(prob_labels_list):
    """
    :param prob_labels_list: list of matrices with probability distribution p(y|x) for every batch
    :return: list with predicted class labels witch data from all batches
    """
    predict = []
    for labels_prob in prob_labels_list:
        predict += predict_labels(labels_prob)
    return predict


def euclidean_distance(x, x_train):
    """
    Euclidean distance (L2) between *X* and *X_train*.
    :param x: what compare [N1xD]
    :param x_train: to what compare [N2xD]
    :return: matrix distances between "X" and "X_train" [N1xN2]
    """
    dot_product = np.dot(x, x_train.T)
    sum_square_test = np.square(x).sum(axis=1)
    sum_square_train = np.square(x_train).sum(axis=1)
    dists = np.sqrt(-2 * dot_product + sum_square_train + np.array([sum_square_test]).T)
    return dists


# In every row labels sorted by distance
def sort_train_labels(dist, y):
    """
    Sort *y* (class labels) by probability from matrix *dist*

    :param dist: matrix of distances between *X* and *X_train* [N1xN2]
    :param y: vector of labels [1xN2]
    :return: matrix of sorted labels, by probability from matrix *dist* [N1xN2]
    """
    result = np.zeros(shape=(dist.shape[0], dist.shape[1]))
    sort_arg = np.argsort(dist, 1, kind="mergesort")
    for vec in range(dist.shape[0]):
        for i in range(len(y)):
            result[vec, i] = y[sort_arg[vec, i]]
    return result


def p_y_x(y, k):
    """
    Calculating probability distribution for every class

    :param y: matrix of sorted labels for training data [N1xN2]
    :param k: amount of nearest neighbours
    :return:  matrix with probability distribution p(y|x) for every class for object from *X*  [N1xM]
    """
    results = []
    y_uniques_count = np.unique(y).shape[0]
    for neighbours in y:
        knn = np.array(neighbours[:k])
        row = [np.count_nonzero(knn == i) / k for i in range(y_uniques_count)]
        results.append(row)
    return np.array(results)


def classification_error(p_y_x, y_true):
    """
    Calculating  classification error (less is better)

    :param p_y_x: predicted probability - every row represent p(y|x) [NxM]
    :param y_true: real labels for class [1xN]
    :return: classification error
    """
    N = len(y_true)
    result = 0.0
    for i in range(N):
        max_prob = 0
        prob_class = 0
        for j in range(p_y_x.shape[1]):
            if p_y_x[i, j] >= max_prob:
                max_prob = p_y_x[i, j]
                prob_class = j
        if prob_class != y_true[i]:
            result += 1
    return result / N


def model_select(x_val, x_train, y_val, y_train, k_values):
    """
    1. Calculating error for different *k* values
    2. Selecting *k* with min error

    :param x_val: validation data  [N1xD]
    :param x_train: training set [N2xD]
    :param y_val: class labels for validation data [1xN1]
    :param y_train: class labels for training data [1xN2]
    :param k_values: searched *k* values
    :return: tuple: (best_error, best_k)
            - *best_error* - minimal error
            - *best_k* - *k* for which minimal error
    """
    distances = euclidean_distance(x_val, x_train)
    sorted_labels = sort_train_labels(distances, y_train)
    best_k = k_values[0]
    best_err = np.inf
    for i in range(np.size(k_values)):
        k = k_values[i]
        pyx = p_y_x(sorted_labels, k)
        error = classification_error(pyx, y_val)
        if best_err > error:
            best_err = error
            best_k = k
    return best_err, best_k


def model_select_with_splitting_to_batches(x_val, x_train, y_val, y_train, k_values, batch_size):
    """
    1. Split training data to minibatches
    2. For every batch select best *k* value with min err parameter
    3. From *k* values which had been found choose best (these with the least min err)

    :return:tuple: (best_error, best_k)
            - *best_error* - minimal error
            - *best_k* - *k* for which minimal error
    """

    best_k = k_values[0]
    best_err = np.inf

    if batch_size < len(x_val):
        x_val_batches = split_to_batches(x_val, batch_size)
        y_val_batches = split_to_batches(y_val, batch_size)
        batches_qty = len(x_val_batches)

        for i in range(batches_qty):
            print('- Searching best k for batch: ', i, "/", batches_qty, sep="")
            err, k = model_select(x_val_batches[i], x_train, y_val_batches[i], y_train, k_values)
            if err < best_err:
                best_err = err
                best_k = k
        return best_err, best_k
    return model_select(x_val, x_train, y_val, y_train, k_values)


def model_select_batches_and_selecting_val_train_proportion(x_train, y_train, k_vals, batch_size, min=5, max=95,
                                                            step=5):
    best_err = np.inf
    best_k = k_vals[0]
    best_val_perc = 0
    for perc in range(min, max, step):
        start_time = time.time()
        print("- Checking proportion (val to train): ", perc, "/100", sep="")
        (new_x_train, new_y_train), (x_val, y_val) = split_to_train_and_val(x_train, y_train, perc)
        err, k = model_select_with_splitting_to_batches(x_val, new_x_train, y_val, new_y_train, k_vals, batch_size)
        if best_err > err:
            best_err = err
            best_k = k
            best_val_perc = perc
        end_time = time.time()
        print("- Done in: ", get_time_result(end_time - start_time))
    (new_best_x_train, new_best_y_train), (_, _) = split_to_train_and_val(x_train, y_train, best_val_perc)
    return best_err, best_k, best_val_perc, new_best_x_train, new_best_y_train


def gen_result_report(k, val_batch_size, test_batch_size, validation_set_percent, correctness, total_time):
    name_msg = "KNeighborsClassifier"
    k_msg = "n_neighbors: " + str(k)
    dist_msg = "distance: " + DISTANCE_CALC_METHOD
    val_msg = "batch size of validation set: " + str(val_batch_size)
    test_msg = "batch size of test set: " + str(test_batch_size)
    proportion = "best validation set size: " + str(validation_set_percent) + "%"
    param_msg = "{ " + k_msg + ", " + dist_msg + "," + proportion + "," + val_msg + ", " + test_msg + " }"
    return [name_msg, param_msg, correctness, total_time]
