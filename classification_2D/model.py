from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet169
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, Input, MaxPool2D
from keras.models import Model

from intracranial_hemorrhage_detection.params import tf_image_size

def get_model():
    "Returns the classifier model"

    input = Input(shape=(tf_image_size, tf_image_size, 1))
    x = BatchNormalization(axis=-1)(input)

    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(1024, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv2D(2048, (3, 3), activation="relu", kernel_initializer="he_uniform")(x)
    x = MaxPool2D()(x)
    x = BatchNormalization(axis=-1)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_initializer="he_uniform")(x)
    x = Dense(1024, activation='linear', kernel_initializer="he_uniform")(x)
    predictions = Dense(6, activation='sigmoid')(x)

    return Model(inputs=input, outputs=predictions)
