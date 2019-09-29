from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.models import Model

from intracranial_hemorrhage_detection.classification_2D.params import input_image_size

def get_model():
    "Returns the classifier model"

    input = Input(shape=(input_image_size, input_image_size, 1))
    x = BatchNormalization(axis=-1)(input)

    x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)

    x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)

    x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(1024, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_initializer="he_uniform")(x)
    x = Dense(1024, activation='linear', kernel_initializer="he_uniform")(x)
    predictions = Dense(6, activation='sigmoid')(x)

    return Model(inputs=input, outputs=predictions)
