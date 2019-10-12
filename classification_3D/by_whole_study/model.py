from keras.layers import Conv3D, Input, Reshape
from keras.models import Model

from intracranial_hemorrhage_detection.classification_3D.by_whole_study.params import input_slice_size, max_slices_per_study
from intracranial_hemorrhage_detection.classification_3D.model_building_blocks import reducing_block, extending_block, final_block

def get_model():
    "Returns the model for voxel classification, inputs the whole image + padding"

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

    assert x.shape[1:] == (max_slices_per_study, 1, 1, 512), "x should be of shape (?, %s, 1, 1, nb_channels), instead it is %s. You can use final_blocks to reduce width and height" % (max_slices_per_study, x.shape)
    predictions = Conv3D(6, (1, 1, 1), padding="same", activation="sigmoid", kernel_initializer="he_uniform")(x)
    predictions = Reshape(target_shape=(max_slices_per_study, 6))(predictions)

    return Model(inputs=input, outputs=predictions)
