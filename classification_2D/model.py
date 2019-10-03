from keras.layers import Conv2D, Dense, Flatten, Input, MaxPool2D
from keras.models import Model

from intracranial_hemorrhage_detection.classification_2D.params import input_image_size

def get_bloc(input, nb_channels):
    x = Conv2D(nb_channels, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(input)
    x = Conv2D(nb_channels, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(nb_channels * 2, (3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)
    return x

def get_model():
    "Returns the classifier model"

    input = Input(shape=(input_image_size, input_image_size, 1))

    x = get_bloc(input, 64)
    x = get_bloc(x, 128)
    x = get_bloc(x, 256)
    x = get_bloc(x, 512)
    x = get_bloc(x, 1024)
    x = get_bloc(x, 2048)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_initializer="he_uniform")(x)
    predictions = Dense(6, activation='sigmoid')(x)

    return Model(inputs=input, outputs=predictions)
