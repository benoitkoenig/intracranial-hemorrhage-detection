# import csv
import os
import pydicom

from intracranial_hemorrhage_detection.constants import folder_path

# Documentation for reading dicom files at https://pydicom.github.io/pydicom/stable/viewing_images.html#using-pydicom-with-matplotlib

def get_all_images_id(folder):
    """
    Loads a list of all images in a given folder name. Returns a list of (id, filepath)\n
    folder can be either 'stage_1_train' or 'stage_1_test'
    """
    assert (folder == "stage_1_train") | (folder == "stage_1_test")

    dirname = folder_path + "/data/%s_images/" % folder
    return [(dirname + filename, filename[:-4]) for filename in os.listdir(dirname)]

def get_dicom_data(file_path):
    "Return the dicom raw data of a given file"
    return pydicom.dcmread(file_path)
