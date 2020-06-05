from IPython.core.display import SVG
from tensorflow.python.keras.utils.vis_utils import plot_model, model_to_dot

from app.utils.data_utils import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

VAL_SIZE = 0.25
RANDOM_STATE = 2046703

# values with best accuracy (todo)
EPOCHS = 25  # 75
BATCH_SIZE = 256  # 2048


# -------------------------------------------------------------------------------------------------------------------- #
def pre_processing_dataset(X_train, y_train, X_test, y_test):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, X_val = scale_data(X_train, X_test, X_val)
    X_train, X_test, X_val = change_data_to_3d(X_train, X_test, X_val)
    return X_train, y_train, X_test, y_test, X_val, y_val


def create_model():
    return keras.Sequential([
        Conv2D(32, (3, 3), input_shape=IMG_SHAPE, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), input_shape=IMG_SHAPE, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])


def compile_model(model):
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def feed_model(model, X_train, y_train, X_val, y_val, batch_size, epochs):
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, y_val),
    )
    return history


def evaluate_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=3)
    return test_loss, test_acc


def make_prediction(model, x_test):
    predictions = model.predict(x_test)
    return predictions


# -------------------------------------------------------------------------------------------------------------------- #
def log_printer(log, path_pref, name, extension="txt"):
    with open(path_pref + name + "." + extension, 'w') as f:
        print(log, file=f)


def plot_history_graphs(history, path_pref, name, extension=".png"):
    plot_accuracy_history(history, path_pref, name + "_accuracy", extension)
    plot_losses_history(history, path_pref, name + "_losses", extension)


def plot_accuracy_history(history, path_pref, name, extension):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_pref + name + extension)
    plt.show()


def plot_losses_history(history, path_pref, name, extension):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(path_pref + name + extension)
    plt.show()


def plot_model_svg(model, path_pref, name):
    plot_model(model, to_file=path_pref + name + ".png")
    SVG(model_to_dot(model).create(prog='dot', format='svg'))
