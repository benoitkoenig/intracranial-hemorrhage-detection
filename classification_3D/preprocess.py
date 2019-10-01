import cv2
import math
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.params import input_slice_size
from intracranial_hemorrhage_detection.constants import max_slices_per_study

def preprocess_voxels(voxels):
    "Inputs a list of voxel. Outputs a numpy array with the right shape"
    number_of_slices = len(voxels)
    voxels = [np.array(pixels, dtype=np.float32) for pixels in voxels]
    voxels= [cv2.resize(pixels, (input_slice_size, input_slice_size)) for pixels in voxels]
    voxels = np.array(voxels, dtype=np.float32)
    voxels -= np.min(voxels)
    voxels /= np.max(voxels)
    voxels = np.reshape(voxels, (number_of_slices, input_slice_size, input_slice_size, 1))
    return voxels

def add_padding(voxels, labels):
    """
    Adds black slices on top and bottom of the voxels and zeros to the labels to end up with a constant height\n
    voxels input shape is (?, input_slice_size, input_slice_size, 1) and outputs shape (max_slices_per_study, input_slice_size, input_slice_size, 1)\n
    labels input shape is (?, 6) and outputs shape (max_slices_per_study, 6)
    """
    assert voxels.shape[0] == labels.shape[0]
    assert voxels.shape[1:] == (input_slice_size, input_slice_size, 1)
    assert labels.shape[1:] == (6,)

    padding = (max_slices_per_study - voxels.shape[0]) / 2
    padding_before = math.ceil(padding)
    padding_after = math.floor(padding)

    zeros_before_voxels = np.zeros((padding_before, input_slice_size, input_slice_size, 1), dtype=np.float32)
    zeros_after_voxels = np.zeros((padding_after, input_slice_size, input_slice_size, 1), dtype=np.float32)
    new_voxels = np.concatenate((zeros_before_voxels, voxels, zeros_after_voxels))

    zeros_before_labels = np.zeros((padding_before, 6), dtype=np.float32)
    zeros_after_labels = np.zeros((padding_after, 6), dtype=np.float32)
    new_labels = np.concatenate((zeros_before_labels, labels, zeros_after_labels))

    return (new_voxels, new_labels)
