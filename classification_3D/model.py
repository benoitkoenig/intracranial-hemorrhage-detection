from keras.layers import Conv3D, Dense, Flatten, Input, MaxPool3D, Reshape
from keras.models import Model

from intracranial_hemorrhage_detection.classification_3D.params import input_slice_size
from intracranial_hemorrhage_detection.constants import max_slices_per_study

def get_model():
    "Returns the classifier model"

    input = Input(shape=(max_slices_per_study, input_slice_size, input_slice_size, 1))
    x = Flatten()(input)
    x = Dense(max_slices_per_study * 6, activation='sigmoid')(x)
    predictions = Reshape(target_shape=(max_slices_per_study, 6))(x)

    return Model(inputs=input, outputs=predictions)
