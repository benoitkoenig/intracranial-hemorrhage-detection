from intracranial_hemorrhage_detection.classification_2D.predict import get_2D_classifier_prediction
from intracranial_hemorrhage_detection.export_predictions.to_csv import clear_outputs_csv, save_image_predictions_to_outputs_csv
from intracranial_hemorrhage_detection.preprocess import get_all_images_list

batch_size = 2000

def export():
    clear_outputs_csv("results_2D")
    test_set = get_all_images_list("stage_1_test")
    while len(test_set) != 0:
        batch = test_set[:batch_size]
        test_set = test_set[batch_size:]
        predictions = [get_2D_classifier_prediction(filepath) for (_, filepath) in batch]
        ids = [id for (id, _) in batch]
        save_image_predictions_to_outputs_csv(ids, predictions, "results_2D")

export()
