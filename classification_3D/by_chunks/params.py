import numpy as np

from intracranial_hemorrhage_detection.constants import folder_path, stage_1_studies_count

chunk_size = 8
batch_size = 4
chunk_wise_data_len = 90497
steps_per_epoch = 100
test_set_size = 497
real_total_epochs = 20

assert (chunk_wise_data_len - test_set_size) % (batch_size * steps_per_epoch) == 0
epochs = real_total_epochs * (chunk_wise_data_len - test_set_size) / (batch_size * steps_per_epoch)

dtype = np.float16

model_weights_path = folder_path + "/weights/classifier_3D_by_chunks.hdf5"
input_slice_size = 256

learning_rate = 1e-4
