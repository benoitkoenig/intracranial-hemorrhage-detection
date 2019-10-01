from keras.layers import Concatenate, Conv3D, Dense, Flatten, Input, MaxPool3D, Reshape, UpSampling3D
from keras.models import Model

from intracranial_hemorrhage_detection.classification_3D.params import input_slice_size
from intracranial_hemorrhage_detection.constants import max_slices_per_study

def get_model():
    "Returns the classifier 3D model"

    input = Input(shape=(max_slices_per_study, input_slice_size, input_slice_size, 1))

    x1 = Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(input)
    x1 = Conv3D(64, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x1)
    x1 = Conv3D(128, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x1)
    x1b = MaxPool3D((1, 16, 16))(x1)

    x2 = MaxPool3D((2, 2, 2))(x1)
    x2 = Conv3D(128, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x2)
    x2 = Conv3D(128, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x2)
    x2 = Conv3D(256, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x2)
    x2b = MaxPool3D((1, 4, 4))(x2)

    x3 = MaxPool3D((2, 2, 2))(x2)
    x3 = Conv3D(256, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x3)
    x3 = Conv3D(256, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x3)
    x3 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x3)

    x4 = MaxPool3D((1, 2, 2))(x3)
    x4 = UpSampling3D((2, 1, 1))(x4)
    x4 = Concatenate(axis=-1)([x2b, x4])
    x4 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x4)
    x4 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x4)
    x4 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x4)

    x5 = MaxPool3D((1, 2, 2))(x4)
    x5 = UpSampling3D((2, 1, 1))(x5)
    x5 = Concatenate(axis=-1)([x1b, x5])
    x5 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x5)
    x5 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x5)
    x5 = Conv3D(512, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x5)

    x6 = Flatten()(x5)
    x6 = Dense(1024, activation='relu')(x6)

    predictions = Dense(max_slices_per_study * 6, activation='sigmoid')(x6)
    predictions = Reshape(target_shape=(max_slices_per_study, 6))(predictions)

    return Model(inputs=input, outputs=predictions)
