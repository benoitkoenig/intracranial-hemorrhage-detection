import pandas as pd
import random

from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image
from intracranial_hemorrhage_detection.classification_3D.params import batch_size
from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.preprocess import get_all_true_labels

def get_study_wise_data():
    df = pd.read_csv("%s/outputs/study_ids.csv" % folder_path)
    grouped_slice_ids = df["slice_ids"].values
    grouped_slice_ids = [eval(i) for i in grouped_slice_ids]
    return grouped_slice_ids

def training_generator():
    "Yields a tuple ([X], [Y]). X is a np array containing the images of shape (batch_size, None, input_image_size, input_image_size, 1) and true_labels is an array of 'shape' (batch_size, None, 6)"
    grouped_slice_ids = get_study_wise_data()
    random.shuffle(grouped_slice_ids)
    true_labels = get_all_true_labels()
    while (len(grouped_slice_ids) != 0):
        batch = grouped_slice_ids[:batch_size]
        grouped_slice_ids = grouped_slice_ids[batch_size:]

        X = [generate_3D_image(["%s/data/stage_1_train_images/%s.dcm" % (folder_path, id) for id in slice_ids]) for slice_ids in batch]
        Y = [[true_labels[id] for id in slice_ids] for slice_ids in batch]
        yield ([X], [Y])
