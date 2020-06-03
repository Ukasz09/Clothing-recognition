import numpy as np


def hamming_distance(X, X_train):
    """
    Between *X* and *X_train*.

    :param X: what compare [N1xD]
    :param X_train: to what compare [N2xD]
    :return: matrix distances between "X" and "X_train" [N1xN2]
    """
    X_arr = X.toarray().astype(int)
    X_arr_trans = np.transpose(X_train.toarray()).astype(int)
    return np.array(X_arr.shape[1] - X_arr @ X_arr_trans - (1 - X_arr) @ (1 - X_arr_trans))


# In every row labels sorted ascending by distance
def sort_train_labels_knn(dist, y):
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


def p_y_x_knn(y, k):
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
    Calculating  classification error

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


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    1. Calculating error for different *k* values
    2. Selecting best *k* using KNN (which mean chose min *k*)

    :param X_val: validation data  [N1xD]
    :param X_train: training set [N2xD]
    :param y_val: class labels for validation data [1xN1]
    :param y_train: class labels for training data [1xN2]
    :param k_values: searched *k* values
    :return: tuple: (best_error, best_k, errors)
            - *best_error* - minimal error
            - *best_k* - *k* for which minimal error
            - *errors* - error list for *k* from *k_values*
    """
    errors = []
    distances = hamming_distance(X_val, X_train)
    sorted_labels = sort_train_labels_knn(distances, y_train)
    best_k = k_values[0]
    best_err = np.inf
    for i in range(np.size(k_values)):
        pyx = p_y_x_knn(sorted_labels, k_values[i])
        error = classification_error(pyx, y_val)
        errors.append(error)
        if best_err > error:
            best_err = error
            best_k = k_values[i]
    return best_err, best_k, np.array(errors)
