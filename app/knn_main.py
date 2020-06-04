from app.data_utils import *
from app.knn_utils import *
from app.prediction_utils import *

PREDICTION_RESULT_CSV = "knn_prediction.csv"
REPORT_RESULT_CSV = "knn_report.csv"

VAL_BATCH_SIZE = 2500
TEST_BATCH_SIZE = 2000


# -------------------------------------------------------------------------------------------------------------------- #
def run_knn_test():
    print('\n------------- KNN model  ')
    print('\n------------- Loading data  ')

    (train_images, train_labels), (test_images, test_labels) = load_scaled_and_flatten_data()

    print('\n------------- Generating k values    ')

    k_list = candidate_k_values()

    print('\n------------- Searching for best k value  \n')

    start_total_time = time.time()
    start_time = time.time()
    best_err, best_k, best_val_perc, train_images, train_labels = model_select_batches_and_selecting_val_train_proportion(
        train_images,
        train_labels, k_list,
        VAL_BATCH_SIZE)
    end_time = time.time()
    print("- Completed in: ", get_time_result(end_time - start_time))

    print('\n------------- Best k has been found ')
    print('\nBest k: {num1} , best error: {num2:.4f}, best validation set size: {num3} %'.format(
        num1=best_k, num2=best_err, num3=best_val_perc))

    print('\n------------- Calculating class labels probability for test data \n')

    start_time = time.time()
    prob_labels_list = predict_prob_with_splitting_to_batches(test_images, train_images, train_labels, best_k,
                                                              TEST_BATCH_SIZE)
    end_time = time.time()
    print("- Completed in: ", get_time_result(end_time - start_time))
    print('\n------------- Choosing most probable labels for test data ')

    predicted_labels = predict_labels_for_every_batch(prob_labels_list)

    print('------------- Saving prediction results to csv file  ')

    save_labels_to_csv(predicted_labels, PREDICTION_RESULT_CSV)

    print('------------- Calculating prediction accuracy  ')

    accuracy = calc_accuracy(predicted_labels, test_labels)
    end_total_time = time.time()

    print('------------- Predicting result  \n')
    total_time = get_time_result(end_total_time - start_total_time)
    print("Predicting accuracy: ", accuracy, "%", ". Total calculation time= ", total_time, sep="")

    print('\n------------- Saving result report to csv file ')
    report_param = gen_result_report(best_k, VAL_BATCH_SIZE, TEST_BATCH_SIZE, best_val_perc, accuracy, total_time)
    save_result_report(report_param, REPORT_RESULT_CSV)
    return prob_labels_list, predicted_labels


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images, test_images = scale_data(train_images, test_images)

    plot_example_images(train_images, train_labels)
    prob_label_list, predicted_labels = run_knn_test()
    plot_image_with_predict_bar(test_images, test_labels, prob_label_list[0], predicted_labels)
    exit(0)
