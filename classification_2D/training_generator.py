import gc
import numpy as np
import random
import tensorflow as tf

from intracranial_hemorrhage_detection.classification_2D.params import subset_size
from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_tf_image, get_all_true_labels

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
    print("\n\n****Warning\nCreate_subset did not return with a full subset. The training will likely interumpted by an error message\n")
    return subset

def training_generator(graph):
    """
    Yields a tuple (image, true_labels). Image is a tensor of shape (1, tf_image_size, tf_image_size, 3) and true_labels is a tensor of shape (1, 6)\n
    Due to the way generators work, it is required to specify the graph to work on\n
    """
    images_list = get_all_images_list("stage_1_train")
    random.shuffle(images_list)
    true_labels = get_all_true_labels()
    subset = create_subset(images_list, true_labels)
    random.shuffle(subset)
    for (id, filepath) in subset:
        with graph.as_default():
            tf_image = get_tf_image(filepath)
            true_label = tf.convert_to_tensor([true_labels[id]], dtype=tf.float32)
        gc.collect()
        yield ([tf_image], [true_label])
