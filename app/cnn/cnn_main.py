from app.utils.prediction_utils import *
from app.cnn.cnn_utils import *
import os
from sys import exit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LABELS_RESULT_PREF = "results/logs/predictions_"
LOG_PREF = "results/logs/log_"

MODEL_PATH_PREF = "results/models/model_"
HISTORY_PATH_PREF = "results/models/history_"

log_text = ""


def log(txt_line):
    global log_text
    log_text += txt_line + "\n"
    print(txt_line)


def clear_log():
    global log_text
    log_text = ""


# -------------------------------------------------------------------------------------------------------------------- #
def run_cnn(test_name, data_loader_func, batch_size=None, epochs=None):
    clear_log()
    log('Test name: ' + test_name + "\n")
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    log('Batch size: ' + str(batch_size))
    log('Epochs: ' + str(epochs))
    print('------------- CNN model  ')
    print('------------- Loading data  ')
    X_train, y_train, X_test, y_test, X_val, y_val = pre_processing_dataset(*data_loader_func())
    print('------------- Creating model  ')
    model = create_model()
    print('------------- Saving model image ')
    plot_model_svg(model, MODEL_PATH_PREF, test_name)
    print('------------- Compiling model  ')
    compile_model(model)
    start_time = time.time()
    print('------------- Training model  ')
    history = feed_model(model, X_train, y_train, X_val, y_val, batch_size, epochs)
    print('------------- Saving fitting history graphs ')
    plot_history_graphs(history, HISTORY_PATH_PREF, test_name)
    print('-------------  Making labels predictions for test data')
    predictions = make_prediction(model, X_test)
    print('------------- Predict labels for test data ')
    predicted_labels = predict_labels(predictions)
    print('------------- Saving prediction results to file  ')
    save_labels_to_csv(predicted_labels, LABELS_RESULT_PREF + test_name)
    print('------------- Evaluate accuracy  ')
    loss, accuracy = evaluate_accuracy(model, X_test, y_test)
    calculating_time = convert_time(time.time() - start_time)
    log('Prediction accuracy: ' + str(round(accuracy * 100, 2)) + '% \n' +
        'Prediction loss: ' + str(round(loss, 2)) + '\n' +
        'Total calculation time: ' + str(calculating_time))
    log_printer(log_text, LOG_PREF, test_name)
    return predictions, predicted_labels


# for tests
def get_debased_data(batch_size=500):
    debased = [split_to_batches(d, batch_size)[0] for d in [*load_normal_data()]]
    return tuple(debased)


# -------------------------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    run_cnn("1", get_debased_data)
    exit(0)
