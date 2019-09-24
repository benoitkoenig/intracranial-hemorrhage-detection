import csv
import pandas as pd

from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_dicom_data

columns = ["study_id", "slice_ids"]

def create_study_wise_data():
    all_files = get_all_images_list("stage_1_train")
    study_ids = {}

    for (slice_id, filepath) in all_files:
        dicom_data = get_dicom_data(filepath)
        study_id = dicom_data["0020", "000d"][0:]
        if (study_id not in study_ids):
            study_ids[study_id] = []
        study_ids[study_id].append(slice_id)

    slice_ids = []
    study_ids_list = []
    for study_id in study_ids:
        study_ids_list.append(study_id)
        slice_ids.append(study_ids[study_id])
    df = pd.DataFrame({ "study_id": study_ids_list, "slice_ids": slice_ids }, columns=columns)
    df.to_csv(folder_path + "/study_ids.csv", header=True, index=False)

create_study_wise_data()
