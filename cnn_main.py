from tensorflow.python.keras.utils.vis_utils import plot_model
from app.utils.prediction_utils import *
from app.cnn.cnn_utils import *
from sys import exit


MODELS_PATH = "app/cnn/results/models/"
LOGS_PATH = "app/cnn/results/logs/"

PREDICTIONS_PREFIX = "predictions_"
LOG_PREFIX = "log_"
MODEL_IMG_PREFIX = "model_"
HISTORY_IMG_PREFIX = "history_"
EXAMPLE_IMG_PREFIX = "example_"

log_text = ""


def log(txt_line):
    global log_text
    log_text += txt_line + "\n"
    print(txt_line)


def clear_logs():
    global log_text
    log_text = ""


# -------------------------------------------------------------------------------------------------------------------- #
def run_cnn(data_loader_func, batch_size=EPOCHS, epochs=BATCH_SIZE, with_augmentation=True):
    test_name = str(USED_MODEL_NUMBER) + '_epoch' + str(EPOCHS) + '_batch' + str(BATCH_SIZE)
    clear_logs()
    print('------------- CNN model  ')
    print('------------- Loading data  ')
    X_train, y_train, X_test, y_test, X_val, y_val = pre_processing_dataset(*data_loader_func())
    log('Test name: ' + test_name + "\n")
    log('Batch size: ' + str(batch_size))
    log('Epochs: ' + str(epochs))
    log('Started data size qty: ' + str(X_train.shape[0]))
    print('------------- Creating model  ')
    cnn_model = create_model()
    print('------------- Saving model image ')
    plot_model(cnn_model, to_file=MODELS_PATH + MODEL_IMG_PREFIX + test_name + ".png")
    print('------------- Compiling model  ')
    compile_model(cnn_model)
    start_time = time.time()
    print('------------- Training model')
    augm_gen_func = augm_gen if with_augmentation else None
    history = feed_model(cnn_model, X_train, y_train, X_val, y_val, batch_size, epochs, augm_gen_func=augm_gen_func)
    print('------------- Saving data fitting history graphs ')
    augm_suffix = "_augm" if with_augmentation else "_norm"
    plot_history_graphs(history, MODELS_PATH, HISTORY_IMG_PREFIX + test_name + augm_suffix)
    print('-------------  Making labels predictions for test data')
    predictions = make_prediction(cnn_model, X_test)
    print('------------- Predicting labels for test data ')
    predicted_labels = predict_labels(predictions)
    print('------------- Saving prediction results to file  ')
    save_labels_to_csv(predicted_labels, LOGS_PATH, PREDICTIONS_PREFIX + test_name)
    print('------------- Evaluating accuracy  ')
    loss, accuracy = evaluate_accuracy(cnn_model, X_test, y_test)
    log('Prediction accuracy: ' + str(round(accuracy * 100, 2)) + '% \n' +
        'Prediction loss: ' + str(round(loss, 2)) + '\n' +
        'Total calculation time: ' + str(convert_time(time.time() - start_time)))
    log_printer(log_text, LOGS_PATH + LOG_PREFIX, test_name)
    return predictions, predicted_labels, test_name


# -------------------------------------------------------------------------------------------------------------------- #
# For quick tests
def get_debased_data(batch_size=200):
    debased = [split_to_batches(d, batch_size)[0] for d in [*load_normal_data()]]
    return tuple(debased)


def plot_examples(predictions, predicted_labels, qty=9):
    X_train, y_train, X_test, y_test, X_val, y_val = pre_processing_dataset(*get_debased_data())
    X_batch, y_batch = augm_gen(X_train).flow(X_train, y_train, batch_size=qty).next()
    plot_rand_images_from_gen(X_batch, y_batch, MODELS_PATH + EXAMPLE_IMG_PREFIX, 'png', color_map='viridis')
    X_test = change_data_to_2d(X_test)
    plot_image_with_predict_bar(X_test, y_test, predictions, predicted_labels, MODELS_PATH + EXAMPLE_IMG_PREFIX, 'png')


if __name__ == "__main__":
    model = create_model()
    predictions, predicted_labels, test_name = run_cnn(load_normal_data)
    model_filepath = MODELS_PATH + MODEL_IMG_PREFIX + test_name + ".png"
    plot_model(model, to_file=model_filepath)
    plot_examples(predictions, predicted_labels)
    exit(0)
