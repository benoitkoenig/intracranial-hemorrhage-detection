from keras.layers import Concatenate, Conv3D, Input, MaxPool3D, Reshape, UpSampling3D
from keras.models import Model

from intracranial_hemorrhage_detection.classification_3D.params import input_slice_size, max_slices_per_study

def reducing_block(input, nb_channels, b_factor, skip_xa=False):
    "The reducing block returns one variable where all dim is divided by 2, and another where depth is total and width+height is fixed to 16"
    x = Conv3D(nb_channels, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(input)
    x = Conv3D(nb_channels, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv3D(2 * nb_channels, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)

    xb = MaxPool3D((1, b_factor, b_factor))(x)
    if (skip_xa == True):
        return (None, xb)
    xa = MaxPool3D((2, 2, 2))(x)
    return (xa, xb)

def extending_block(input1, input2, nb_channels):
    "The extending block upsamples the depth of input1 to match the shape of input2. Width and height are therfore fixed to 16, but depth restores"
    x1 = UpSampling3D((2, 1, 1))(input1)
    x = Concatenate(axis=-1)([x1, input2])
    x = Conv3D(nb_channels, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv3D(nb_channels, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv3D(nb_channels, (3, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    return x

def final_block(input, nb_channels):
    "The final aims at reducing the width and height of the variable, which depth is already max_slices_per_study. The convolutions along depth have f=1"
    x = Conv3D(nb_channels, (1, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(input)
    x = Conv3D(nb_channels, (1, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Conv3D(nb_channels, (1, 3, 3), padding="same", activation="relu", kernel_initializer="he_uniform")(x)
    x = Concatenate(axis=-1)([x, input])
    x = MaxPool3D((1, 2, 2))(x)
    return x

def get_model():
    "Returns the classifier 3D model"

    # Comments track the shape given max_slices_per_study=64 and input_slice_size=256
    input = Input(shape=(max_slices_per_study, input_slice_size, input_slice_size, 1)) # 64x256x256

    (x, x1) = reducing_block(input, 64, 16) # 32x128x128, 64x16x16
    (x, x2) = reducing_block(x, 128, 8) # 16x64x64, 32x16x16
    (x, x3) = reducing_block(x, 256, 4) # 8x32x32, 16x16x16
    (_, x) = reducing_block(x, 512, 2, skip_xa=True) # _, 8x16x16

    x = extending_block(x, x3, 256) # 16x16x16
    x = extending_block(x, x2, 128) # 32x16x16
    x = extending_block(x, x1, 64) # 64x16x16

    # Number of final_blocks depends on input_slice_size. The output should have a width and height of 1
    x = final_block(x, 64) # 64x8x8
    x = final_block(x, 128) # 64x4x4
    x = final_block(x, 256) # 64x2x2

    assert x.shape[1:] == (max_slices_per_study, 1, 1, 256), "x should be of shape (?, %s, 1, 1, nb_channels), instead it is %s. You can use final_blocks to reduce width and height" % (max_slices_per_study, x.shape)
    predictions = Conv3D(6, (1, 1, 1), padding="same", activation="sigmoid", kernel_initializer="he_uniform")(x)
    predictions = Reshape(target_shape=(max_slices_per_study, 6))(predictions)

    return Model(inputs=input, outputs=predictions)
