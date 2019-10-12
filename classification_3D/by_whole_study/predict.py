from keras.models import load_model
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.by_whole_study.params import dtype, model_weights_path
from intracranial_hemorrhage_detection.classification_3D.by_whole_study.padding import add_padding, remove_padding
from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image

model = load_model(model_weights_path, compile=False)

def get_3D_classifier_prediction(slices_ids):
    "Returns the prediction as a numpy array of the same length as labels"
    X = generate_3D_image(slices_ids, folder="stage_1_test")
    X = np.expand_dims(add_padding(X), 0)

    predictions = model.predict(X, steps=1)
    predictions = remove_padding(predictions[0], len(slices_ids))
    return predictions
