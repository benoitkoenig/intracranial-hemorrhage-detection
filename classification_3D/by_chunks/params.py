import numpy as np

from intracranial_hemorrhage_detection.constants import folder_path, stage_1_studies_count

chunk_size = 8
batch_size = 8
chunk_wise_data_len = 1180
steps_per_epoch = 2
epochs = chunk_wise_data_len // (steps_per_epoch * batch_size)
dtype = np.float16

model_weights_path = folder_path + "/weights/classifier_3D_by_chunks.hdf5"
input_slice_size = 256

learning_rate = 1e-5
