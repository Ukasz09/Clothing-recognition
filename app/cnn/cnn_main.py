from app.utils.prediction_utils import *
from app.cnn.cnn_utils import *
import os
from sys import exit

# to remove keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batches_to_check = [32, 50, 64, 128, 256]

# -------------------------------------------------------------------------------------------------------------------- #
LABELS_RESULT_PREF = "results/logs/predictions_"
LOG_PREF = "results/logs/log"

MODEL_PATH_PREF = "results/models/model_"
HISTORY_PATH_PREF = "results/models/history_"
RAND_IMG_PREF = "results/models/example2_"
PREDICTED_IMG_BAR_PREF = "results/models/example_"
MODEL_NUMBER = '1'
log_text = ""


def log(txt_line):
    global log_text
    log_text += txt_line + "\n"
    print(txt_line)


def clear_log():
    global log_text
    log_text = ""


# -------------------------------------------------------------------------------------------------------------------- #
def run_cnn(test_name, data_loader_func, batch_size=None, epochs=None, with_augmentation=True):
    clear_log()
    print('------------- CNN model  ')
    print('------------- Loading data  ')
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    X_train, y_train, X_test, y_test, X_val, y_val = pre_processing_dataset(*data_loader_func())
    log('Test name: ' + test_name + "\n")
    log('Batch size: ' + str(batch_size))
    log('Epochs: ' + str(epochs))
    log('Started data size qty: ' + str(X_train.shape[0]))
    print('------------- Creating model  ')
    model = create_model()
    print('------------- Saving model image ')
    plot_model_svg(model, MODEL_PATH_PREF, test_name)
    print('------------- Compiling model  ')
    compile_model(model)
    start_time = time.time()
    if with_augmentation:
        print('------------- Training model on augmented data')
        history = feed_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, augm_gen)
        print('------------- Saving augmented data fitting history graphs ')
        plot_history_graphs(history, HISTORY_PATH_PREF, test_name + "_augm")
    else:
        print('------------- Training model on normal data  ')
        history = feed_model(model, X_train, y_train, X_val, y_val, batch_size, epochs)
        print('------------- Saving normal data fitting history graphs ')
        plot_history_graphs(history, HISTORY_PATH_PREF, test_name + "_norm")
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


# -------------------------------------------------------------------------------------------------------------------- #
# for tests
def get_debased_data(batch_size=200):
    debased = [split_to_batches(d, batch_size)[0] for d in [*load_normal_data()]]
    return tuple(debased)


def plot_examples(predictions, predicted_labels, qty=9):
    X_train, y_train, X_test, y_test, X_val, y_val = pre_processing_dataset(*get_debased_data())
    X_batch, y_batch = augm_gen(X_train).flow(X_train, y_train, batch_size=qty).next()
    plot_rand_images_from_gen(X_batch, y_batch, RAND_IMG_PREF, 'png', plt_show=True, color_map='viridis')
    X_test = change_data_to_2d(X_test)
    plot_image_with_predict_bar(X_test, y_test, predictions, predicted_labels, PREDICTED_IMG_BAR_PREF, 'png',
                                plt_show=True)


if __name__ == "__main__":
    model = create_model()
    test_name = str(MODEL_NUMBER) + '_epoch' + str(EPOCHS) + '_batch' + str(BATCH_SIZE)
    plot_model_svg(model, MODEL_PATH_PREF, test_name)
    # predictions, predicted_labels = run_cnn(test_name, get_debased_data)
    # plot_examples(predictions, predicted_labels)
    exit(0)
