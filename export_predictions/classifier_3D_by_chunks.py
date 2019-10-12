import math

from intracranial_hemorrhage_detection.classification_3D.preprocess import get_study_wise_data
from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import chunk_size
from intracranial_hemorrhage_detection.classification_3D.by_chunks.predict import get_3D_classifier_prediction
from intracranial_hemorrhage_detection.export_predictions.to_csv import clear_outputs_csv, save_image_predictions_to_outputs_csv

step_size = chunk_size // 2

def split_ids_into_chunks(slice_ids):
    i = 0
    chunks = []
    while i * step_size + chunk_size < len(slice_ids):
        chunks.append(slice_ids[i * step_size : i * step_size + chunk_size])
        i += 1
    chunks.append(slice_ids[-chunk_size:])
    return chunks

def restore_predictions_from_chunks(predictions, total_size):
    bottom = chunk_size // 4
    top = step_size + chunk_size // 4
    preds = predictions[0][:top]
    for i in range(len(predictions) - 2):
        preds += predictions[i + 1][bottom:top]
    missing_preds_count = total_size - len(preds)
    missing_preds = predictions[-1][-missing_preds_count:]
    preds += missing_preds
    return preds

def export():
    clear_outputs_csv("results_3D_by_chunks")
    grouped_slice_ids = get_study_wise_data("stage_1_test")
    for slice_ids in grouped_slice_ids:
        chunks = split_ids_into_chunks(slice_ids)
        predictions = [get_3D_classifier_prediction(chunk) for chunk in chunks]
        predictions = [p.tolist() for p in predictions]
        predictions = restore_predictions_from_chunks(predictions, len(slice_ids))

        save_image_predictions_to_outputs_csv(slice_ids, predictions, "results_3D_by_chunks")

if __name__ == "__main__":
    export()
