import csv
import cv2
import numpy as np
import os
import pydicom

from intracranial_hemorrhage_detection.constants import folder_path, label_ranks

def get_all_images_list(folder):
    """
    Loads a list of all images in a given folder name. Returns a list of (id, filepath)\n
    folder can be either 'stage_1_train' or 'stage_1_test'
    """
    assert folder in ["stage_1_train", "stage_1_test"]

    dirname = folder_path + "/data/%s_images/" % folder
    return [(filename[:-4], dirname + filename) for filename in os.listdir(dirname)]

def get_dicom_data(filepath):
    "Return the dicom raw data of a given file"
    return pydicom.dcmread(filepath)

def get_input_images(filepaths, input_image_size):
    """
    Inputs a list of the filepaths to the dicom files and the size to which images must be resized\n
    Returns the images as resized numpy arrays. Output shape: (len(filepaths), input_image_size, input_image_size, 1)
    """
    output = []
    for filepath in filepaths:
        dicom_data = get_dicom_data(filepath)
        image = np.array(dicom_data.pixel_array, dtype=np.float32)
        image = cv2.resize(image, (input_image_size, input_image_size))
        image -= np.min(image)
        image /= np.max(image)
        image = 2 * image - 1
        image = np.reshape(image, (input_image_size, input_image_size, 1))
        output.append(image)
    return np.array(output)

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
