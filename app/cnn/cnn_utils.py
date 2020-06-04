from app.utils.data_utils import *
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

TEST_SIZE = 0.2
RANDOM_STATE = 2020
EPOCHS = 75
BATCH_SIZE = 4096


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
        MaxPooling2D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def compile_model(model):
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def feed_model(model, X_train, y_train, X_val, y_val):
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_val, y_val),
    )


def evaluate_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_loss, test_acc


def make_prediction(model, x_test):
    predictions = model.predict(x_test)
    return predictions
