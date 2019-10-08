import pandas as pd

from intracranial_hemorrhage_detection.constants import folder_path, labels

columns = ["ID", "Label"]

def clear_outputs_csv(filename):
    "Resets the file {filename}.csv: clear all the lines if the file exists, create the file otherwise"
    df = pd.DataFrame({}, columns=columns)
    df.to_csv("%s/outputs/%s.csv" % (folder_path, filename), header=True, index=False)

def save_image_predictions_to_outputs_csv(image_ids, predictions, filename):
    "Save the predictions to {filename}.csv with the right ids"
    ids = [["%s_%s" % (image_id, label) for label in labels] for image_id in image_ids]
    df = pd.DataFrame({
        "ID": [item for sublist in ids for item in sublist],
        "Label": [item for sublist in predictions for item in sublist],
    }, columns=columns)
    df.to_csv("%s/outputs/%s.csv" % (folder_path, filename), mode="a", header=False, index=False)
