import math
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.by_whole_study.params import max_slices_per_study
from intracranial_hemorrhage_detection.constants import folder_path

def add_padding(input):
    "For a np array of any shape, adds 0s along axis=0 so that the first dimension is max_slices_per_study"
    padding = (max_slices_per_study - input.shape[0]) / 2
    zeros_before = np.zeros([math.ceil(padding)] + list(input.shape[1:]))
    zeros_after = np.zeros([math.floor(padding)] + list(input.shape[1:]))
    return np.concatenate((zeros_before, input, zeros_after))

def remove_padding(predictions, output_len):
    padding = (len(predictions) - output_len) / 2
    padding_before = math.ceil(padding)
    padding_after = math.floor(padding)
    return predictions[padding_before : len(predictions) - padding_after]
