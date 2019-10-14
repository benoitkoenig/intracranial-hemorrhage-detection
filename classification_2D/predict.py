from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import model_weights_path
from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.classification_2D.preprocess import get_input_images

model = get_model()
model.load_weights(model_weights_path)

def get_2D_classifier_prediction(filepath):
    "Returns the prediction as a numpy array of the same length as labels"
    X = get_input_images([filepath])
    predictions = model.predict(X, steps=1)
    return predictions[0]
