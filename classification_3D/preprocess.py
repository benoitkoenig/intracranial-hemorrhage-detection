import cv2
import math
import numpy as np
import pandas as pd

from intracranial_hemorrhage_detection.constants import folder_path

def preprocess_voxels(voxels, dtype, input_slice_size):
    "Inputs a list of voxel. Outputs a numpy array with the right shape"
    number_of_slices = len(voxels)
    voxels = [np.array(pixels, dtype=np.float32) for pixels in voxels] # must use float32 for cv2.resize, even if we use another dtype later
    voxels= [cv2.resize(pixels, (input_slice_size, input_slice_size)) for pixels in voxels]
    voxels = np.array(voxels, dtype=dtype)
    voxels -= np.min(voxels)
    voxels /= np.max(voxels)
    voxels = np.reshape(voxels, (number_of_slices, input_slice_size, input_slice_size, 1))
    return voxels

def get_study_wise_data(folder):
    "Returns the sudy-wise data as stored in the corresponding csv file. Folder should be one of 'stage_1_train', 'stage_1_test'"
    assert (folder in ["stage_1_train", "stage_1_test"])
    df = pd.read_csv("%s/outputs/study_ids_%s.csv" % (folder_path, folder))
    grouped_slice_ids = df["slice_ids"].values
    grouped_slice_ids = [eval(i) for i in grouped_slice_ids]
    return grouped_slice_ids
