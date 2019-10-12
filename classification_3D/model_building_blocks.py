from keras.layers import Concatenate, Conv3D, MaxPool3D, UpSampling3D

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
