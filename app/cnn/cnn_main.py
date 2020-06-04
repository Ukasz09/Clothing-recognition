from app.utils.prediction_utils import *
from app.cnn.cnn_utils import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PREDICTION_RESULT_CSV = "csv/cnn_prediction.csv"
REPORT_RESULT_CSV = "csv/cnn_report.csv"


def run_cnn():
    print('\n------------- CNN model  ')
    print('\n------------- Loading data  ')
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = pre_processing_dataset()

    print('------------- Creating model  ')
    model = create_model()
    print('------------- Compiling model  ')
    compile_model(model)

    # todo: tmp data
    tmp_data = get_debased_data(X_train, y_train, X_val, y_val, X_test, y_test)
    #

    start_time = time.time()
    print('------------- Training model  ')
    feed_model(model, tmp_data[0], tmp_data[1], tmp_data[2], tmp_data[3])
    print('------------- Calculating class labels probability for test data')
    predictions = make_prediction(model, tmp_data[4])
    print('------------- Choosing most probable labels for test data ')
    predicted_labels = predict_labels(predictions)
    print('------------- Saving prediction results to csv file  ')
    save_labels_to_csv(predicted_labels, PREDICTION_RESULT_CSV)
    print('------------- Calculating prediction accuracy  ')
    accuracy = calc_accuracy(predicted_labels, tmp_data[5])

    calculating_time = convert_time(time.time() - start_time)
    print("Predicting accuracy: ", accuracy, "%", ". Total calculation time= ", calculating_time, sep="")
    # evaluate_accuracy(model, X_test, y_test)
    return predictions, predicted_labels


# For tests
def get_debased_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=500):
    data = [X_train, y_train, X_val, y_val, X_test, y_test]
    return [split_to_batches(d, batch_size)[0] for d in data]


if __name__ == "__main__":
    run_cnn()
    exit(0)
