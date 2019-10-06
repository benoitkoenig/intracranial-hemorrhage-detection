from keras.models import load_model
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image
from intracranial_hemorrhage_detection.classification_3D.params import dtype, model_weights_path
from intracranial_hemorrhage_detection.classification_3D.postprocess import remove_padding
from intracranial_hemorrhage_detection.classification_3D.preprocess import add_padding

model = load_model(model_weights_path, compile=False)

def get_3D_classifier_prediction(slices_ids):
    "Returns the prediction as a numpy array of the same length as labels"
    X = generate_3D_image(slices_ids, folder="stage_1_test")
    X = np.expand_dims(add_padding(X), 0)

    predictions = model.predict(X, steps=1)
    predictions = remove_padding(predictions[0], len(slices_ids))
    return predictions
