from app.data_utils import *
from app.knn_utils import *
from app.prediction_utils import *

PREDICTION_RESULT_CSV = "knn_prediction.csv"
REPORT_RESULT_CSV = "knn_report.csv"

VAL_BATCH_SIZE = 500  # 2500
TEST_BATCH_SIZE = 500  # 2000


# -------------------------------------------------------------------------------------------------------------------- #
def run_knn_test():
    print('\n-------------   KNN model  -------------')
    print('\n-------------   Loading data  -------------')

    (train_images, train_labels), (test_images, test_labels) = load_scaled_and_flatten_data()
    (train_images, train_labels), (val_images, val_labels) = split_to_train_and_val(train_images, train_labels)

    print('\n-------------   Generating k values    -------------')

    k_list = candidate_k_values()

    print('\n------------- Searching for best k value  -------------\n')

    start_total_time = time.time()
    start_time = time.time()

    # todo: tmp
    val_img_tmp = split_to_batches(val_images, 500)[0]
    val_lab_tmp = split_to_batches(val_labels, 500)[0]

    # best_err, best_k = model_select_with_splitting_to_batches(val_images, train_images, val_labels, train_labels,
    #                                                           k_list)
    best_err, best_k = model_select_with_splitting_to_batches(val_img_tmp, train_images, val_lab_tmp, train_labels,
                                                              range(1, 15, 1), batch_size=VAL_BATCH_SIZE)
    end_time = time.time()
    print("- Completed in: ", get_time_result(end_time - start_time))

    print('\n------------- Best k has been found -------------')
    print('\nBest k: {num1} , best error: {num2:.4f}'.format(num1=best_k, num2=best_err))
    print('\n------------- Calculating class labels probability for test data -------------')

    start_time = time.time()
    prob_labels_list = predict_prob_with_splitting_to_batches(split_to_batches(test_images, batch_size=500)[0],
                                                              train_images, train_labels, best_k, TEST_BATCH_SIZE)
    end_time = time.time()
    print("- Completed in: ", get_time_result(end_time - start_time))
    print('\n------------- Choosing most probable labels for test data -------------')

    start_time = time.time()
    predicted_labels = predict_labels_for_every_batch(prob_labels_list)
    end_time = time.time()
    print("- Completed in: ", get_time_result(end_time - start_time))

    print('\n------------- Saving prediction results to csv file  -------------')

    save_labels_to_csv(predicted_labels, PREDICTION_RESULT_CSV)

    print('\n------------- Calculating prediction accuracy  -------------')

    accuracy = calc_accuracy(predicted_labels, split_to_batches(test_labels, batch_size=500)[0])

    end_total_time = time.time()

    print('\n------------- Predicting result  -------------\n')
    total_time = get_time_result(end_total_time - start_total_time)
    print("Predicting accuracy: ", accuracy, "%", ". Total calculation time= ", total_time, sep="")

    print('\n------------- Saving result report to csv file -------------\n')
    report_param = gen_result_report(best_k, VAL_BATCH_SIZE, TEST_BATCH_SIZE, accuracy, total_time)
    save_result_report(report_param, REPORT_RESULT_CSV)


def display_images_template():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images, test_images = scale_data(train_images, test_images)
    display_example_images(train_images, train_labels)


if __name__ == "__main__":
    display_images_template()
    # run_knn_test()
    exit(0)
