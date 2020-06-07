from app.utils.data_utils import *
from app.knn.knn_utils import *
from app.utils.prediction_utils import *
import os
from sys import exit

VAL_SIZE = 0.25
TEST_BATCH_SIZE = 2000
BEST_K = 7


# -------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------- #
def run_knn_test(val_size=VAL_SIZE, k=BEST_K):
    print('\n------------- KNN model - predicting  ')
    print('------------- Loading data  ')
    X_train, y_train, X_test, y_test = pre_processing_dataset()
    (X_train, y_train), (_, _) = split_to_train_and_val(X_train, y_train, val_size)
    start_total_time = time.time()
    print('------------- Making labels predictions for test data')
    start_time = time.time()
    predictions_list = predict_prob_with_splitting_to_batches(X_test, X_train, y_train, k, TEST_BATCH_SIZE)
    print("- Completed in: ", convert_time(time.time() - start_time))
    print('\n------------- Predicting labels for test data')
    predicted_labels = predict_labels_for_every_batch(predictions_list)
    print('------------- Saving prediction results to file')
    save_labels_to_csv(predicted_labels, PREDICTION_RESULT_CSV_PREF + "_k" + str(k))
    print('------------- Evaluating accuracy ')
    accuracy = calc_accuracy(predicted_labels, y_test)
    print('------------- Saving prediction results to file  ')
    print('------------- Results ')
    clear_log(CALCULATING_ACCURACY_PREF + str(k))
    log("KNN\n", CALCULATING_ACCURACY_PREF + str(k))
    log('Distance calc algorithm: ' + DISTANCE_CALC_METHOD, CALCULATING_ACCURACY_PREF + str(k))
    log('k: ' + str(k), CALCULATING_ACCURACY_PREF + str(k))
    log('Train images qty: ' + str(X_train.shape[0]), CALCULATING_ACCURACY_PREF + str(k))
    log('Accuracy: ' + str(accuracy) + '%\nTotal calculation time= '
        + str(convert_time(time.time() - start_total_time)), CALCULATING_ACCURACY_PREF + str(k))
    print('\n------------- Result saved to file ')
    return predictions_list, predicted_labels


def select_best_k(X_train, y_train, val_size=VAL_SIZE, batch_size=2500):
    print('------------- Searching for best k value')
    start_time = time.time()
    (X_train, y_train), (X_val, y_val) = split_to_train_and_val(X_train, y_train, val_size)
    err, k = model_select_with_splitting_to_batches(X_val, X_train, y_val, y_train, candidate_k_values(), batch_size)
    end_time = time.time()
    calc_time = convert_time(end_time - start_time)
    clear_log(K_VALUE_SEARCHING_LOG_PREF + str(k))
    print('------------- Best k has been found ')
    log('One batch size: ' + str(batch_size), K_VALUE_SEARCHING_LOG_PREF)
    log('Train images qty: ' + str(X_train.shape[0]), K_VALUE_SEARCHING_LOG_PREF)
    log('Validation images qty: ' + str(X_val.shape[0]), K_VALUE_SEARCHING_LOG_PREF)
    log('Distance calc algorithm: ' + DISTANCE_CALC_METHOD, K_VALUE_SEARCHING_LOG_PREF)
    log('Best k: ' + str(k) + '\nBest error: ' + str(err) + "\nCalculation time: " + str(calc_time),
        K_VALUE_SEARCHING_LOG_PREF)
    return k


def get_debased_data(batch_size=500):
    debased = [split_to_batches(d, batch_size)[0] for d in [*pre_processing_dataset()]]
    return tuple(debased)


def plot_examples(predictions, predicted_labels):
    X_train, y_train, X_test, y_test = load_normal_data()
    X_train, X_test = scale_x(X_train, X_test)
    plot_rand_images(X_train, y_train, RAND_IMG_PREF, 'png', plt_show=True)
    plot_image_with_predict_bar(X_test, y_test, predictions, predicted_labels, PREDICTED_IMG_BAR_PREF, 'png',
                                plt_show=True)


# todo: test
def plot_examples():
    X_train, y_train, X_test, y_test = load_normal_data()
    X_train, X_test = scale_x(X_train, X_test)
    plot_rand_images(X_train, y_train, RAND_IMG_PREF, 'png', plt_show=True)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = pre_processing_dataset()
    best_k = select_best_k(X_train, y_train)
    predictions_list, predicted_labels = run_knn_test(k=best_k)
    # plot_examples(predictions_list[0], predicted_labels)
    # plot_examples()
    exit(0)
