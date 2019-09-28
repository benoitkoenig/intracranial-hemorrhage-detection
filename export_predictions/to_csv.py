import pandas as pd

from intracranial_hemorrhage_detection.constants import folder_path, labels

export_file_path = "%s/outputs/results.csv" % folder_path
columns = ["ID", "Label"]

def clear_outputs_csv():
    "Resets the file results.csv: clear all the lines if the file exists, create the file otherwise"
    df = pd.DataFrame({}, columns=columns)
    df.to_csv(export_file_path, header=True, index=False)

def save_image_predictions_to_outputs_csv(image_id, predictions):
    "Save the predictions to results.csv with the right ids"
    df = pd.DataFrame({
        "ID": ["%s_%s" % (image_id, label) for label in labels],
        "Label": predictions,
    }, columns=columns)
    df.to_csv(export_file_path, mode="a", header=False, index=False)
