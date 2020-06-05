from app.utils.data_utils import *
from app.knn.knn_utils import *
from app.utils.prediction_utils import *

PREDICTION_RESULT_CSV = "results/knn_prediction.results"
REPORT_RESULT_CSV = "results/knn_report.results"

VAL_BATCH_SIZE = 2500
TEST_BATCH_SIZE = 2000


# -------------------------------------------------------------------------------------------------------------------- #
def run_knn_test(best_val_perc=25, best_k=5):
    print('\n------------- KNN model  ')
    print('\n------------- Loading data  ')
    (X_train, y_train), (X_test, y_test) = pre_procesing_dataset()
    (X_train, y_train), (_, _) = split_to_train_and_val(X_train, y_train, best_val_perc)

    print('\n------------- Calculating class labels probability for test data \n')
    start_time = time.time()
    prediction_list = predict_prob_with_splitting_to_batches(X_test, X_train, y_train, best_k, TEST_BATCH_SIZE)
    calculating_time = convert_time(time.time() - start_time)
    print("- Completed in: ", calculating_time)
    print('\n------------- Choosing most probable labels for test data ')
    predicted_labels = predict_labels_for_every_batch(prediction_list)
    print('------------- Saving prediction results to results file  ')
    save_labels_to_csv(predicted_labels, PREDICTION_RESULT_CSV)
    print('------------- Calculating prediction accuracy  ')
    accuracy = calc_accuracy(predicted_labels, y_test)
    print("Predicting accuracy: ", accuracy, "%", ". Total calculation time= ", calculating_time, sep="")
    print('\n------------- Saving result report to results file ')
    report_param = gen_result_report(best_k, VAL_BATCH_SIZE, TEST_BATCH_SIZE, best_val_perc, accuracy, calculating_time)
    save_report_to_csv(report_param, REPORT_RESULT_CSV)
    return prediction_list, predicted_labels


# time consuming
def select_best_k_and_val_proportion(x_train, y_train, k_vals, batch_size, min=5, max=95,
                                     step=5):
    print('\n------------- Searching for best k value  \n')
    start_total_time = time.time()

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
        print("- Done in: ", convert_time(time.time() - start_time))
    print("- Completed in: ", convert_time(time.time() - start_total_time))
    print('\n------------- Best k has been found ')
    print('\nBest k: {num1} , best error: {num2:.4f}, best validation set size: {num3} %'.format(num1=best_k,
                                                                                                 num2=best_err,
                                                                                                 num3=best_val_perc))
    return best_err, best_k, best_val_perc


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_normal_data()
    train_images, test_images = scale_data(train_images, test_images)

    prob_label_list, predicted_labels = run_knn_test()
    plot_rand_images(train_images, train_labels)
    plot_image_with_predict_bar(test_images, test_labels, prob_label_list[0], predicted_labels, True)
    exit(0)
