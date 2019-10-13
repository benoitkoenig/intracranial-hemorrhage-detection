import efficientnet.keras as efn
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from intracranial_hemorrhage_detection.classification_2D.params import input_image_size

def get_model():
    "Returns a classifier model"
    efficient_net = efn.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(input_image_size, input_image_size, 3),
    )

    x = efficient_net.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=efficient_net.input, outputs=predictions)

    return model
