import numpy as np
import random

from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image
from intracranial_hemorrhage_detection.classification_3D.params import batch_size
from intracranial_hemorrhage_detection.classification_3D.preprocess import add_padding, get_study_wise_data
from intracranial_hemorrhage_detection.preprocess import get_all_true_labels

def training_generator():
    """
    Yields a tuple ([X], [Y]).\n
    X is a np array containing the images of shape (batch_size, max_slices_per_study, input_image_size, input_image_size, 1)
    and true_labels is a np array of shape (batch_size, max_slices_per_study, 6)
    """
    grouped_slice_ids = get_study_wise_data("stage_1_train")
    random.shuffle(grouped_slice_ids)
    true_labels = get_all_true_labels()
    while (len(grouped_slice_ids) != 0):
        batch = grouped_slice_ids[:batch_size]
        grouped_slice_ids = grouped_slice_ids[batch_size:]

        X = [generate_3D_image(slice_ids) for slice_ids in batch]
        Y = [np.array([true_labels[id] for id in slice_ids], dtype=np.float32) for slice_ids in batch]
        X = np.array([add_padding(x) for x in X])
        Y = np.array([add_padding(y) for y in Y])
        yield ([X], [Y])
