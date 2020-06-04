from app.utils.data_utils import *
from app.utils.prediction_utils import *
from app.cnn.cnn_utils import *


def run_cnn():
    (train_images, train_labels), (test_images, test_labels) = load_data()
    train_images, test_images = scale_data(train_images, test_images)
    #
    model = create_model()
    compile_model(model)
    feed_model(model, train_images, train_labels)
    # predictions = make_prediction(model, test_images)
    # predicted_labels = predict_labels(predictions)
    # accuracy = calc_accuracy(predicted_labels, test_labels)
    # print("Accuracy:", accuracy)
    loss, acc = evaluate_accuracy(model, test_images, test_labels)
    print(loss, acc)


if __name__ == "__main__":
    run_cnn()
    exit(0)
