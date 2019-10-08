from intracranial_hemorrhage_detection.classification_3D.preprocess import get_study_wise_data
from intracranial_hemorrhage_detection.classification_3D.predict import get_3D_classifier_prediction
from intracranial_hemorrhage_detection.export_predictions.to_csv import clear_outputs_csv, save_image_predictions_to_outputs_csv

def export():
    clear_outputs_csv("results_3D")
    grouped_slice_ids = get_study_wise_data("stage_1_test")
    for slice_ids in grouped_slice_ids:
        predictions = get_3D_classifier_prediction(slice_ids)
        save_image_predictions_to_outputs_csv(slice_ids, predictions, "results_3D")

export()
