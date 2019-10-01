from intracranial_hemorrhage_detection.classification_2D.predict import get_2D_classifier_prediction
from intracranial_hemorrhage_detection.export_predictions.to_csv import clear_outputs_csv, save_image_predictions_to_outputs_csv
from intracranial_hemorrhage_detection.preprocess import get_all_images_list

def export():
    clear_outputs_csv("results_2D")
    test_set = get_all_images_list("stage_1_test")
    for (id, filepath) in test_set:
        predictions = get_2D_classifier_prediction(filepath)
        save_image_predictions_to_outputs_csv(id, predictions, "results_2D")

export()
