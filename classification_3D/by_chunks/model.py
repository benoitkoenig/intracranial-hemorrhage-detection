from keras.layers import Conv3D, Input, Reshape
from keras.models import Model

from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import chunk_size, input_slice_size
from intracranial_hemorrhage_detection.classification_3D.model_building_blocks import reducing_block, extending_block, final_block

def get_model():
    "Returns the model for voxel classification, inputs the whole image + padding"

    # Comments track the shape given chunk_size=8 and input_slice_size=256
    input = Input(shape=(chunk_size, input_slice_size, input_slice_size, 1)) # 8x256x256

    (x, x1) = reducing_block(input, 64, 8) # 4x128x128, 8x32x32
    (x, x2) = reducing_block(x, 128, 4) # 2x64x64, 4x32x32
    (x, x3) = reducing_block(x, 256, 2) # 1x32x32, 2x32x32
    (_, x) = reducing_block(x, 512, 1, skip_xa=True) # _, 1x32x32

    x = extending_block(x, x3, 256) # 2x32x32
    x = extending_block(x, x2, 128) # 4x32x32
    x = extending_block(x, x1, 64) # 8x32x32

    # Number of final_blocks depends on input_slice_size. The output should have a width and height of 1
    x = final_block(x, 64) # 8x16x16
    x = final_block(x, 128) # 8x8x8
    x = final_block(x, 256) # 8x4x4
    x = final_block(x, 512) # 8x2x2
    x = final_block(x, 1024) # 8x1x1

    assert x.shape[1:] == (chunk_size, 1, 1, 2048), "x should be of shape (?, %s, 1, 1, nb_channels), instead it is %s. You can use final_blocks to reduce width and height" % (chunk_size, x.shape)
    predictions = Conv3D(6, (1, 1, 1), padding="same", activation="sigmoid", kernel_initializer="he_uniform")(x)
    predictions = Reshape(target_shape=(chunk_size, 6))(predictions)

    return Model(inputs=input, outputs=predictions)
