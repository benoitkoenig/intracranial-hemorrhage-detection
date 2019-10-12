from keras.models import load_model
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import dtype, input_slice_size, model_weights_path
from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image

model = load_model(model_weights_path, compile=False)

def get_3D_classifier_prediction(chunk):
    "Returns the prediction as a numpy array of the same length as chunk"
    X = generate_3D_image(chunk, dtype, input_slice_size, folder="stage_1_test")
    X = np.expand_dims(X, 0)

    predictions = model.predict(X, steps=1)
    return predictions[0]
