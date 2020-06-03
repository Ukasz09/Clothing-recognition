from app.data_utils import *
from app.knn_utils import *
from app.prediction_utils import *


def gen_rand_k(min_k, max_k, step=2):
    return range(min_k, max_k, step)


def run_knn_test():
    print('\n-------------   KNN model  -------------')
    print('\n-------------   Loading data  -------------')
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images, test_images = scale_data(train_images, test_images)
    train_images = flat_matrix(train_images)
    test_images = flat_matrix(test_images)
    print('\n-------------   Spliting data to batches  -------------')
    train_image_batches = split_to_batches(train_images)
    train_label_batches = split_to_batches(train_labels)
    print('\n-------------   Generating k values  -------------')
    # todo: tmp -> change to 1 : sqrt(len+1)
    k_values = range(1, 101, 2)
    print('\n------------- Searching for best k  -------------\n')
    # best_err, best_k = model_selection_knn(train_image_batches[0], train_images, train_label_batches[0],
    #                                        train_labels, k_values)

    # todo: do func for choosing best k from all batches

    # best_err2, best_k2 = model_selection_knn(train_image_batches[0], train_image_batches[1], train_label_batches[0], train_label_batches[1], k_values)
    # if best_err2 < best_err:
    #     best_k = best_k2
    #     best_err=best_err2
    print('\n------------- Best k has been found -------------')
    # print('\nBest k: {num1} , best error: {num2:.4f}'.format(num1=best_k, num2=best_err))
    print('\n------------- Calculating labels probability for test data -------------')
    test_batches = split_to_batches(test_images, 200)
    labels_probability1 = predict_probability(test_batches[0], train_images, train_labels, 1)
    # labels_probability2 = predict_probability(test_batches[1], train_images, train_labels, best_k)
    # labels_probability3 = predict_probability(test_batches[2], train_images, train_labels, best_k)
    # labels_probability4 = predict_probability(test_batches[3], train_images, train_labels, best_k)
    # labels_probability5 = predict_probability(test_batches[4], train_images, train_labels, best_k)
    print('\n------------- Choosing most probable labels probability for test data -------------')
    predicted_labels1 = get_predicted_labels(labels_probability1)
    # predicted_labels2 = get_predicted_labels(labels_probability2)
    # predicted_labels3 = get_predicted_labels(labels_probability3)
    # predicted_labels4 = get_predicted_labels(labels_probability4)
    # predicted_labels5 = get_predicted_labels(labels_probability5)
    predicted_labels = predicted_labels1  # + predicted_labels2 + predicted_labels3 + predicted_labels4 + predicted_labels5
    print('\n------------- Calculating prediction correctness  -------------')
    correctness1 = calc_correctness(predicted_labels, split_to_batches(test_labels, batch_size=200)[0])
    print('\n------------- Predicting result  -------------\n')
    print(correctness1, "%", sep="")


if __name__ == "__main__":
    run_knn_test()
