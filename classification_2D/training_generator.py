import numpy as np
import random
import tensorflow as tf

from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_tf_image, get_all_true_labels

def training_generator(graph):
    """
    Yields a tuple (image, true_labels). Image is a tensor of shape (1, tf_image_size, tf_image_size, 3) and true_labels is a tensor of shape (1, 6)\n
    Due to the way generators work, it is required to specify the graph to work on\n
    """
    images_list = get_all_images_list("stage_1_train")
    true_labels = get_all_true_labels()
    for (id, filepath) in images_list:
        with graph.as_default():
            tf_image = get_tf_image(filepath)
            true_label = tf.convert_to_tensor([true_labels[id]], dtype=tf.float32)
        yield ([tf_image], [true_label])
