from app.cnn.models import available_models
from app.utils.data_utils import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

VAL_SIZE = 0.25
RANDOM_STATE = 2046703
EPOCHS = 15
BATCH_SIZE = 32

USED_MODEL_NUMBER = 3


# -------------------------------------------------------------------------------------------------------------------- #
def pre_processing_dataset(X_train, y_train, X_test, y_test):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, X_val = scale_all_x(X_train, X_test, X_val)
    X_train, X_test, X_val = change_data_to_3d(X_train, X_test, X_val)
    return X_train, y_train, X_test, y_test, X_val, y_val


def create_model():
    return available_models[USED_MODEL_NUMBER]


def compile_model(model):
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def feed_model(model, X_train, y_train, X_val, y_val, batch_size, epochs, augm_gen_func=None):
    if augm_gen_func is None:
        return fit_normal(model, X_train, y_train, X_val, y_val, batch_size, epochs)
    return fit_with_gen(model, X_train, y_train, X_val, y_val, batch_size, epochs, augm_gen_func)


def fit_normal(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val)
    )
    return history


def fit_with_gen(model, X_train, y_train, X_val, y_val, batch_size, epochs, augm_gen):
    history = model.fit(
        augm_gen(X_train).flow(X_train, y_train, batch_size=batch_size),
        validation_data=(augm_gen(X_val).flow(X_val, y_val, batch_size=batch_size)),
        steps_per_epoch=math.ceil(len(X_train) / batch_size), epochs=epochs,
        validation_steps=math.ceil(len(X_val) / batch_size))
    return history


def evaluate_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=3)
    return test_loss, test_acc


def make_prediction(model, x_test):
    predictions = model.predict(x_test)
    return predictions


# -------------------------------------------------------------------------------------------------------------------- #
def plot_history_graphs(history, path_prefix, filename, extension=".png"):
    plot_accuracy_history(history, path_prefix, filename, extension)
    plot_losses_history(history, path_prefix, filename, extension)


def plot_accuracy_history(history, path_prefix, filename, extension):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy: ' + filename)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_prefix + filename + "_accuracy" + extension)
    plt.show()


def plot_losses_history(history, path_prefix, filename, extension):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss: ' + filename)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_prefix + filename + "_losses" + extension)
    plt.show()
