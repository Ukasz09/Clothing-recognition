from app.utils.data_utils import *
from app.utils.prediction_utils import *

LOGS_PATH = "app/knn/results/logs/"
K_SEARCHING_TXT_PREFIX = "k_searching_"

distances_name = {
    0: "Manhattan_distance",
    1: "Hamming distance",
    2: "Euclidean distance"
}

distances_methods = {
    0: lambda x, x_tr: manhattan_distance(x, x_tr),
    1: lambda x, x_tr: hamming_distance(x, x_tr),
    2: lambda x, x_tr: euclidean_distance(x, x_tr)
}

used_distance_number = 2
distance_name = distances_name[used_distance_number]


# -------------------------------------------------------------------------------------------------------------------- #
def log(text, path, with_print=True):
    if with_print:
        print(text)
    log_printer(text, path, None, append=True)


def clear_log_file(path):
    with open(path + '.txt', 'w+') as f:
        f.truncate(0)


# -------------------------------------------------------------------------------------------------------------------- #
def pre_processing_dataset():
    X_train, y_train, X_test, y_test = load_normal_data()
    X_train, X_test = scale_x(X_train, X_test)
    X_train = flat(X_train)
    X_test = flat(X_test)
    return X_train, y_train, X_test, y_test


def candidate_k_values(min_k=1, max_k=25, step=1):
    """
    :return: list of candidates k values to check
    """
    return range(min_k, max_k, step)


def predict_prob(X_test, X_train, y_train, k):
    """
    :param X_test: test data matrix [N1xW]
    :param X_train: training data matrix [N2xW]
    :param y_train: real class labels for *x_train* object [N2X1]
    :param k: amount of nearest neighbours
    :return: matrix with probability distribution p(y|x) for every class and *x_test* object [N1xM]
    """
    distances = distances_methods[used_distance_number](X_test, X_train)
    sorted_labels = sort_train_labels(distances, y_train)
    return p_y_x(sorted_labels, k)


def predict_prob_with_batches(X_test, X_train, y_train, k, batch_size):
    """
    Split *x_test* to batches and for each one calc matrix with probability distribution p(y|x) for every y class
    :param k: amount of nearest neighbours
    :return: list of matrices with probability distribution p(y|x) for every x_test batch
    """
    if batch_size < len(X_test):
        test_batches = split_to_batches(X_test, batch_size)
        batches_qty = len(test_batches)
        y_prob = [predict_prob(test_batches[i], X_train, y_train, k) for i in range(batches_qty)]
        return y_prob
    return [predict_prob(X_test, X_train, y_train, k)]


def predict_labels_for_every_batch(prob_labels_list):
    """
    :param prob_labels_list: list of matrices with probability distribution p(y|x) for every batch
    :return: list with predicted class labels witch data from all batches
    """
    predict = []
    for labels_prob in prob_labels_list:
        predict += predict_labels(labels_prob)
    return predict


def manhattan_distance(x, x_train):
    """
    :param x: what compare [N1xD]
    :param x_train: to what compare [N2xD]
    :return: matrix distances between "X" and "X_train" [N1xN2]
    """
    return np.abs(x[:, 0, None] - x_train[:, 0]) + np.abs(x[:, 1, None] - x_train[:, 1])


def hamming_distance(X, X_train):
    """
    :param X: what compare [N1xD]
    :param X_train: to what compare [N2xD]
    :return: matrix distances between "X" and "X_train" [N1xN2]
    """
    X_arr = X.astype(int)
    X_arr_trans = np.transpose(X_train).astype(int)
    return np.array(X_arr.shape[1] - X_arr @ X_arr_trans - (1 - X_arr) @ (1 - X_arr_trans))


def euclidean_distance(X, X_train):
    """
    :param X: what compare [N1xD]
    :param X_train: to what compare [N2xD]
    :return: matrix distances between "X" and "X_train" [N1xN2]
    """
    dot_product = np.dot(X, X_train.T)
    sum_square_test = np.square(X).sum(axis=1)
    sum_square_train = np.square(X_train).sum(axis=1)
    return np.sqrt(-2 * dot_product + sum_square_train + np.array([sum_square_test]).T)


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


def model_select(X_val, X_train, y_val, y_train, k_values):
    """
    1. Calculating error for different *k* values
    2. Selecting *k* with min error

    :param X_val: validation data  [N1xD]
    :param X_train: training set [N2xD]
    :param y_val: class labels for validation data [1xN1]
    :param y_train: class labels for training data [1xN2]
    :param k_values: searched *k* values
    :return: tuple: (best_error, best_k)
            - *best_error* - minimal error
            - *best_k* - *k* for which minimal error
    """
    distances = distances_methods[used_distance_number](X_val, X_train)
    sorted_labels = sort_train_labels(distances, y_train)
    best_k = k_values[0]
    best_err = np.inf
    for i in range(np.size(k_values)):
        print("Checking k=", k_values[i])
        k = k_values[i]
        pyx = p_y_x(sorted_labels, k)
        error = classification_error(pyx, y_val)
        if best_err > error:
            best_err = error
            best_k = k
    return best_err, best_k


def model_select_with_splitting_to_batches(X_val, X_train, y_val, y_train, k_values, batch_size):
    """
    1. Split training data to minibatches
    2. For every batch select best *k* value with min err parameter
    3. From *k* values which had been found choose best (these with the least min err)

    :return:tuple: (best_error, best_k)
            - *best_error* - minimal error
            - *best_k* - *k* for which minimal error
    """

    k_dict = {}
    err_dict = {}
    if batch_size < len(X_val):
        x_val_batches = split_to_batches(X_val, batch_size)
        y_val_batches = split_to_batches(y_val, batch_size)
        batches_qty = len(x_val_batches)

        log_text = ""
        for i in range(batches_qty):
            txt = 'Searching best k for batch: ' + str(i + 1) + '/' + str(batches_qty)
            log_text += txt
            print(txt)
            start_time = time.time()
            err, k = model_select(x_val_batches[i], X_train, y_val_batches[i], y_train, k_values)
            k_dict[k] = k_dict.get(k, 0) + 1
            err_dict[k] = err_dict.get(k, 0) + err
            end_time = time.time()
            txt = "Done in: " + convert_time(end_time - start_time)
            log_text += txt
            print(txt)
        best_k = max(k_dict, key=k_dict.get)
        best_err = err_dict[best_k] / k_dict[best_k]
        log(log_text, LOGS_PATH + K_SEARCHING_TXT_PREFIX + str(best_k), with_print=False)
        return best_err, best_k
    return model_select(X_val, X_train, y_val, y_train, k_values)
