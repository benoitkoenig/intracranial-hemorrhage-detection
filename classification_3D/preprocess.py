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

def add_padding(input):
    "For a np array of any shape, adds 0s along axis=0 so that the first dimension is max_slices_per_study"
    padding = (max_slices_per_study - input.shape[0]) / 2
    zeros_before = np.zeros([math.ceil(padding)] + list(input.shape[1:]))
    zeros_after = np.zeros([math.floor(padding)] + list(input.shape[1:]))
    return np.concatenate((zeros_before, input, zeros_after))
