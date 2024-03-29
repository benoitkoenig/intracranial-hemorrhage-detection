import csv
import os
import pydicom

from intracranial_hemorrhage_detection.constants import folder_path, label_ranks

def get_all_images_list(folder):
    """
    Loads a list of all images in a given folder name. Returns a list of (id, filepath)\n
    folder can be either 'stage_1_train' or 'stage_1_test'
    """
    assert folder in ["stage_1_train", "stage_1_test", "stage_2_test"]

    dirname = folder_path + "/data/%s_images/" % folder
    return [(filename[:-4], dirname + filename) for filename in os.listdir(dirname)]

def get_dicom_data(filepath):
    "Return the dicom raw data of a given file"
    return pydicom.dcmread(filepath)

def get_all_true_labels():
    "Returns a dict of all labels for all images"
    all_true_labels = {}
    # The csv data is stored in a cache. This way, the csv is read only once
    with open(folder_path + '/data/stage_1_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None) # skip the header
        for row in csv_reader:
            id = row[0][:12]
            label_id = label_ranks[row[0][13:]]
            if (id not in all_true_labels):
                all_true_labels[id] = [0, 0, 0, 0, 0, 0]
            all_true_labels[id][label_id] = int(row[1])

    return all_true_labels
