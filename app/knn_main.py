from app.data_utils import *
from app.knn_utils import *
from app.prediction_utils import *


def run_knn_test():
    print('\n-------------   KNN model  -------------')
    print('\n-------------   Loading data  -------------')

    (train_images, train_labels), (test_images, test_labels) = load_scaled_and_flatten_data()

    print('\n-------------   Generating k values    -------------')

    k_values = candidate_k_values()

    print('\n------------- Searching for best k value  -------------\n')

    best_err, best_k = model_select_with_splitting_to_batches(train_images, train_labels, k_values)

    print('\n------------- Best k has been found -------------')
    print('\nBest k: {num1} , best error: {num2:.4f}'.format(num1=best_k, num2=best_err))
    print('\n------------- Calculating class labels probability for test data -------------')

    prob_labels_list = predict_prob_with_splitting_to_batches(test_images, train_images, train_labels, best_k)

    print('\n------------- Choosing most probable labels for test data -------------')

    predicted_labels = predict_labels(prob_labels_list)

    print('\n------------- Calculating prediction correctness  -------------')

    correctness = calc_correctness(predicted_labels, test_labels)

    print('\n------------- Predicting result  -------------\n')
    print(correctness, "%", sep="")


if __name__ == "__main__":
    run_knn_test()
