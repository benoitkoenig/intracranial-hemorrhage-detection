import numpy as np

from intracranial_hemorrhage_detection.constants import folder_path, stage_1_studies_count

max_slices_per_study = 64
input_slice_size = 128
batch_size = 1 # Unfortunately 64 slices is already very memory consumming, so we will keep batch_size at 1
steps_per_epoch=16
epochs = stage_1_studies_count // (steps_per_epoch * batch_size)

model_weights_path = folder_path + "/weights/classifier_3D_by_whole_study.hdf5"
dtype = np.float16

learning_rate = 1e-5
