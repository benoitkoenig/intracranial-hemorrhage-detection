from keras.layers import Add, Conv3D, Input, Reshape, UpSampling3D
from keras.models import Model

from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import chunk_size, input_slice_size

def get_model():
    "Returns the model for voxel classification, inputs the whole image + padding"

    # Comments track the shape given chunk_size=8 and input_slice_size=256
    input = Input(shape=(chunk_size, input_slice_size, input_slice_size, 1)) # 8x256x256

    x1 = Conv3D(32, (3, 3, 3), strides=1, padding="same", activation="relu", kernel_initializer="he_uniform")(input)
    x1 = Conv3D(64, (3, 3, 3), strides=2, padding="same", activation="relu", kernel_initializer="he_uniform")(x1) # 4x128x128x64

    x2 = Conv3D(64, (3, 3, 3), strides=1, padding="same", activation="relu", kernel_initializer="he_uniform")(x1)
    x2 = Conv3D(128, (3, 3, 3), strides=2, padding="same", activation="relu", kernel_initializer="he_uniform")(x2) # 2x64x64x128

    x3 = Conv3D(128, (3, 3, 3), strides=1, padding="same", activation="relu", kernel_initializer="he_uniform")(x2)
    x3 = Conv3D(256, (3, 3, 3), strides=2, padding="same", activation="relu", kernel_initializer="he_uniform")(x3) # 1x32x32x256

    x1 = Conv3D(128, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x1) # 4x64x64x128
    x1 = Conv3D(256, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x1) # 4x32x32x256
    x2 = Conv3D(256, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x2) # 2x32x32x256

    x = UpSampling3D((2, 1, 1))(x3) # 2x32x32x256
    x = Add()([x, x2])
    x = Conv3D(256, (1, 3, 3), strides=1, padding="same", activation="relu", kernel_initializer="he_uniform")(x)

    x = UpSampling3D((2, 1, 1))(x) # 4x32x32x256
    x = Add()([x, x1])
    x = Conv3D(256, (1, 3, 3), strides=1, padding="same", activation="relu", kernel_initializer="he_uniform")(x)

    x = UpSampling3D((2, 1, 1))(x) # 8x32x32x256
    x = Conv3D(256, (1, 3, 3), strides=1, padding="same", activation="relu", kernel_initializer="he_uniform")(x)

    # Number of final_blocks depends on input_slice_size. The output should have a width and height of 1
    x = Conv3D(512, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x) # 8x16x16
    x = Conv3D(512, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x) # 8x8x8
    x = Conv3D(512, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x) # 8x4x4
    x = Conv3D(512, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x) # 8x2x2
    x = Conv3D(512, (1, 3, 3), strides=(1, 2, 2), padding="same", activation="relu", kernel_initializer="he_uniform")(x) # 8x1x1

    assert x.shape[1:4] == (chunk_size, 1, 1), "x should be of shape (?, %s, 1, 1, nb_channels), instead it is %s. You can use final_blocks to reduce width and height" % (chunk_size, x.shape)
    predictions = Conv3D(6, (1, 1, 1), padding="same", activation="sigmoid", kernel_initializer="he_uniform")(x)
    predictions = Reshape(target_shape=(chunk_size, 6))(predictions)

    return Model(inputs=input, outputs=predictions)
