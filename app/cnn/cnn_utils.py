from app.utils.data_utils import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

TEST_SIZE = 0.2
RANDOM_STATE = 2020


# -------------------------------------------------------------------------------------------------------------------- #
def pre_processing_dataset():
    (X_train, y_train), (X_test, y_test) = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test, X_val = scale_data(X_train, X_test, X_val)
    X_train, X_test, X_val = change_data_to_3d(X_train, X_test, X_val)
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


def create_model():
    model = keras.Sequential([
        Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=IMG_SHAPE),
        MaxPooling2D(pool_size=2),  # down sampling the output instead of 28*28 it is 14*14
        Dropout(0.2),
        Flatten(),  # flatten out the layers
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')

    ])
    return model


def compile_model(model):
    adam = Adam(lr=0.0001, decay=1e-6)
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def feed_model(model, x_train, y_train, x_val, y_val):
    model.fit(
        x_train,
        y_train,
        batch_size=4096,
        epochs=75,
        verbose=1,
        validation_data=(x_val, y_val),
    )


def evaluate_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_loss, test_acc


def make_prediction(model, x_test):
    predictions = model.predict(x_test)
    return predictions


def swish(x, beta=1):
    return x * sigmoid(beta * x)


def add_swish_as_custom():
    get_custom_objects().update({'swish': Activation(swish)})
