import random

from intracranial_hemorrhage_detection.classification_2D.params import batch_size, half_test_size, half_train_size
from intracranial_hemorrhage_detection.classification_2D.preprocess import get_input_images
from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_all_true_labels

def separate_images_any_and_none(images_list, true_labels):
    "Separate images_list into two lists: the first one contains all images with any=1, the second contains all images with any=0"
    any_list = []
    none_list = []
    for (id, filepath) in images_list:
        if (true_labels[id][5] == 1):
            any_list.append((id, filepath))
        else:
            none_list.append((id, filepath))
    return (any_list, none_list)

def training_generator():
    """
    First yields the test set. Then starts yielding batches\n
    Yields a tuple ([X], [Y]). X is a np array containing the images of shape (batch_size, input_image_size, input_image_size, 1) and true_labels is an array of 'shape' (batch_size, 6)
    """
    images_list = get_all_images_list("stage_1_train")
    random.shuffle(images_list)
    true_labels = get_all_true_labels()

    (any_list, none_list) = separate_images_any_and_none(images_list, true_labels)

    any_test_set = any_list[:half_test_size]
    none_test_set = none_list[:half_test_size]
    test_set = any_test_set + none_test_set
    random.shuffle(test_set)
    X_test = get_input_images([filepath for (_, filepath) in test_set])
    Y_test = [true_labels[id] for (id, _) in test_set]
    yield ([X_test], [Y_test])

    any_train_set = any_list[half_test_size:]
    none_train_set = none_list[half_test_size:]

    if (half_train_size != len(any_train_set)):
        print("***Warning: Parametrization failed: any_train_set should be of size %s, but it is actually %s. This should only happen during development" % (half_train_size, len(any_list)))

    subset = []
    while (True):
        if (len(subset) == 0):
            random.shuffle(none_train_set)
            subset = any_train_set + none_train_set[:half_train_size]
            random.shuffle(subset)
        batch = subset[0:batch_size]
        subset = subset[batch_size:]
        X = get_input_images([filepath for (_, filepath) in batch])
        Y = [true_labels[id] for (id, _) in batch]
        yield ([X], [Y])
