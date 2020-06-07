from sys import exit
from app.knn.knn_utils import *
from app.utils.prediction_utils import *

MODELS_PATH = "app/knn/results/models/"

EXAMPLE_IMG_PREFIX = "example_"
PREDICT_CSV_PREFIX = "knn_predictions_"
ACCURACY_TXT_PREFIX = "accuracy_k"

VAL_SIZE = 0.25
BATCH_SIZE = 2500
BEST_K = 7


# -------------------------------------------------------------------------------------------------------------------- #
def run_knn_test(val_size=VAL_SIZE, k=BEST_K):
    print('\n------------- KNN model - predicting  ')
    print('------------- Loading data  ')
    X_train, y_train, X_test, y_test = pre_processing_dataset()
    (X_train, y_train), (_, _) = split_to_train_and_val(X_train, y_train, val_size)
    start_total_time = time.time()
    print('------------- Making labels predictions for test data')
    start_time = time.time()
    predictions_list = predict_prob_with_batches(X_test, X_train, y_train, k, BATCH_SIZE)
    print("- Completed in: ", convert_time(time.time() - start_time))
    print('\n------------- Predicting labels for test data')
    predicted_labels = predict_labels_for_every_batch(predictions_list)
    print('------------- Saving prediction results to file')
    save_labels_to_csv(predicted_labels, LOGS_PATH,  PREDICT_CSV_PREFIX + distance_name + "_k" + str(k))
    print('------------- Evaluating accuracy ')
    accuracy = calc_accuracy(predicted_labels, y_test)
    print('------------- Saving prediction results to file  ')
    print('------------- Results ')
    accuracy_file_path = LOGS_PATH + ACCURACY_TXT_PREFIX + str(k) + '_' + distances_name[used_distance_number]
    clear_log_file(accuracy_file_path)
    log("KNN\n", accuracy_file_path)
    log('Distance calc algorithm: ' + distance_name, accuracy_file_path)
    log('k: ' + str(k), accuracy_file_path)
    log('Train images qty: ' + str(X_train.shape[0]), accuracy_file_path)
    log('Accuracy: ' + str(accuracy) + '%\nTotal calculation time= ' + str(
        convert_time(time.time() - start_total_time)), accuracy_file_path)
    print('\n------------- Result saved to file ')
    return predictions_list, predicted_labels


def select_best_k(X_train, y_train, val_size=VAL_SIZE, batch_size=BATCH_SIZE):
    print('------------- Searching for best k value')
    start_time = time.time()
    (X_train, y_train), (X_val, y_val) = split_to_train_and_val(X_train, y_train, val_size)
    err, k = model_select_with_splitting_to_batches(X_val, X_train, y_val, y_train, candidate_k_values(), batch_size)
    calc_time = convert_time(time.time() - start_time)
    k_searching_path = LOGS_PATH + K_SEARCHING_TXT_PREFIX + str(k)
    clear_log_file(k_searching_path)
    print('------------- Best k has been found ')
    log('One batch size: ' + str(batch_size), k_searching_path)
    log('Train images qty: ' + str(X_train.shape[0]), k_searching_path)
    log('Validation images qty: ' + str(X_val.shape[0]), k_searching_path)
    log('Distance calc algorithm: ' + distance_name, k_searching_path)
    log('Best k: ' + str(k) + '\nBest error: ' + str(err) + "\nCalculation time: " + str(calc_time), k_searching_path)
    return k


# For quick tests
def get_debased_data(batch_size=500):
    return tuple([split_to_batches(d, batch_size)[0] for d in [*pre_processing_dataset()]])


def plot_examples(predictions, predicted_labels):
    X_train, y_train, X_test, y_test = load_normal_data()
    X_train, X_test = scale_x(X_train, X_test)
    image_path = MODELS_PATH + EXAMPLE_IMG_PREFIX
    plot_rand_images(X_train, y_train, image_path, 'png')
    plot_image_with_predict_bar(X_test, y_test, predictions, predicted_labels, image_path, 'png')


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = pre_processing_dataset()
    best_k = select_best_k(X_train, y_train)
    predictions_list, predicted_labels = run_knn_test(k=best_k)
    plot_examples(predictions_list[0], predicted_labels)
    exit(0)
