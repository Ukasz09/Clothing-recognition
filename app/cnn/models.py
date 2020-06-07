from tensorflow import keras
from tensorflow.keras.layers import *
from app.utils.data_utils import IMG_SHAPE
import os

# to remove keras warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -------------------------------------------------------------------------------------------------------------------- #

# Don't forget to add new model also to *available_models* dict

model_1 = keras.Sequential([
    Conv2D(32, (3, 3), input_shape=IMG_SHAPE, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), input_shape=IMG_SHAPE, activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model_2 = keras.Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=IMG_SHAPE),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

model_3 = keras.Sequential([
    Conv2D(32, 3, activation='relu', input_shape=IMG_SHAPE),
    BatchNormalization(),
    Conv2D(64, 3, activation='relu', input_shape=IMG_SHAPE),
    BatchNormalization(),
    MaxPool2D(),
    Conv2D(128, 3, activation='relu', input_shape=IMG_SHAPE),
    BatchNormalization(),
    Conv2D(256, 3, activation='relu', input_shape=IMG_SHAPE),
    BatchNormalization(),
    MaxPool2D(),
    Flatten(),
    Dense(256),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# -------------------------------------------------------------------------------------------------------------------- #
available_models = {
    1: model_1,
    2: model_2,
    3: model_3
}
