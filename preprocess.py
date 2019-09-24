import csv
import os
import pydicom
import tensorflow as tf

from intracranial_hemorrhage_detection.constants import folder_path, label_ranks
from intracranial_hemorrhage_detection.params import tf_image_size

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

def get_tf_image(filepath):
    "Given a filepath, return the image as a resized tensor"
    dicom_data = get_dicom_data(filepath)
    image = tf.convert_to_tensor(dicom_data.pixel_array, dtype=tf.float32)
    image = tf.reshape(image, (1, image.shape[0], image.shape[1], 1))
    image = tf.image.resize(image, (tf_image_size, tf_image_size), align_corners=True, method=tf.image.ResizeMethod.AREA)
    return image

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
