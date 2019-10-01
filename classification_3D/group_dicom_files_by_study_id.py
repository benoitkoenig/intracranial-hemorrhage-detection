import csv
import pandas as pd
import sys

from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_dicom_data

columns = ["study_id", "slice_ids"]

def group_dicom_files_by_study_id(folder):
    all_files = get_all_images_list(folder)
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
    df.to_csv("%s/outputs/study_ids_%s.csv" % (folder_path, folder), header=True, index=False)

# Read arguments from python command
folder = None
for param in sys.argv:
    if param in ["stage_1_train", "stage_1_test"]:
        folder = param

if (folder == None):
    print("Usage: python classification_3D.group_dicom_files_by_study_id.py [stage_1_train|stage_1_test]")
    exit(-1)

group_dicom_files_by_study_id(folder)
