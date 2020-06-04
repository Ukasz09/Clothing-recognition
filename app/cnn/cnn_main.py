from app.utils.prediction_utils import *
from app.cnn.cnn_utils import *


def run_cnn():
    (X_train, y_train), (X_test, y_test), (X_val, y_val) = pre_processing_dataset()
    model = create_model()
    compile_model(model)

    # feed_model(model, X_train, y_train, X_val, y_val)
    # predictions = make_prediction(model, X_test)
    # predicted_labels = predict_labels(predictions)
    # accuracy = calc_accuracy(predicted_labels, y_test)

    tmp_data = get_debased_data(X_train, y_train, X_val, y_val, X_test, y_test)
    feed_model(model, tmp_data[0], tmp_data[1], tmp_data[2], tmp_data[3])
    predictions = make_prediction(model, tmp_data[4])
    predicted_labels = predict_labels(predictions)
    accuracy = calc_accuracy(predicted_labels, tmp_data[5])

    print("Accuracy:", accuracy)


def get_debased_data(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=500):
    data = [X_train, y_train, X_val, y_val, X_test, y_test]
    return [split_to_batches(d, batch_size)[0] for d in data]


if __name__ == "__main__":
    run_cnn()
    exit(0)
