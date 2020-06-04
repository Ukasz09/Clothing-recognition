import tensorflow as tf
from tensorflow import keras
import app.knn.knn_utils as knn


def create_model(in_shape=28):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(in_shape, in_shape)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),
        keras.layers.Softmax()
    ])
    return model


def compile_model(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


def feed_model(model, x_train, y_train):
    model.fit(x_train, y_train, epochs=10)


def evaluate_accuracy(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return test_loss, test_acc

def make_prediction(model, x_test):
    predictions = model.predict(x_test)
    return predictions


def predict_labels(predictions):
    # todo tmp
    return knn.predict_labels(predictions)
