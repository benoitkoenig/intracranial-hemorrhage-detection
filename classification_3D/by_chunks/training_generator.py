import numpy as np
import random

from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image
from intracranial_hemorrhage_detection.classification_3D.by_chunks.preprocess import get_chunk_wise_data
from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import batch_size, chunk_wise_data_len ,dtype, input_slice_size, test_set_size
from intracranial_hemorrhage_detection.preprocess import get_all_true_labels

def training_generator():
    """
    Yields a tuple ([X], [Y]).\n
    X is a np array containing the images of shape (batch_size, chunk_size, input_image_size, input_image_size, 1)
    and true_labels is a np array of shape (batch_size, chunk_size, 6)
    """
    chunks = get_chunk_wise_data("stage_1_train")
    assert len(chunks) == chunk_wise_data_len, "Len of chunks is %s, expected %s" % (len(chunks), chunk_wise_data_len)

    random.shuffle(chunks)
    true_labels = get_all_true_labels()

    test_set = chunks[:test_set_size]
    train_set = chunks[test_set_size:]

    X_test = np.array([generate_3D_image(slice_ids, dtype, input_slice_size) for slice_ids in test_set], dtype=dtype)
    Y_test = np.array([[true_labels[id] for id in slice_ids] for slice_ids in test_set], dtype=dtype)
    yield ([X_test], [Y_test])

    subset = []
    while (True):
        if (len(subset) == 0):
            subset = train_set[0:]
            random.shuffle(subset)
        batch = subset[:batch_size]
        subset = subset[batch_size:]

        X = np.array([generate_3D_image(slice_ids, dtype, input_slice_size) for slice_ids in batch], dtype=dtype)
        Y = np.array([[true_labels[id] for id in slice_ids] for slice_ids in batch], dtype=dtype)
        yield ([X], [Y])
