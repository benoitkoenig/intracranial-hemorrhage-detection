from intracranial_hemorrhage_detection.constants import folder_path, stage_1_studies_count

input_slice_size = 128
batch_size = 4
steps_per_epoch=4
epochs = stage_1_studies_count // (steps_per_epoch * batch_size)

model_weights_path = folder_path + "/weights/classifier_3D.hdf5"
