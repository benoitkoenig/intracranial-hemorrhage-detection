import random

from intracranial_hemorrhage_detection.classification_2D.params import subset_size, batch_size
from intracranial_hemorrhage_detection.classification_2D.preprocess import get_input_images
from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_all_true_labels

def create_subset(images_list, true_labels):
    "Returns a subset with as many images with an hemorrhagge as without. Subset_size is defined in the params"
    subset = []
    any_count = 0
    none_count = 0
    for (id, filepath) in images_list:
        if (true_labels[id][5] == 1) & (any_count < subset_size / 2):
            subset.append((id, filepath))
            any_count += 1
        elif (true_labels[id][5] == 0) & (none_count < subset_size / 2):
            subset.append((id, filepath))
            none_count += 1
        if (any_count == subset_size // 2) & (none_count == subset_size // 2):
            assert len(subset) == subset_size
            return subset
    print("\n\n****Warning\nCreate_subset did not return with a full subset. The training will likely be interumpted by an error message\n")
    return subset

def training_generator():
    "Yields a tuple ([X], [Y]). X is a np array containing the images of shape (batch_size, input_image_size, input_image_size, 1) and true_labels is an array of 'shape' (batch_size, 6)"
    images_list = get_all_images_list("stage_1_train")
    random.shuffle(images_list)
    true_labels = get_all_true_labels()
    subset = create_subset(images_list, true_labels)
    random.shuffle(subset)
    while (len(subset) != 0):
        batch = subset[0:batch_size]
        subset = subset[batch_size:]
        X = get_input_images([filepath for (_, filepath) in batch])
        Y = [true_labels[id] for (id, _) in batch]
        yield ([X], [Y])
