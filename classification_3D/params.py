from intracranial_hemorrhage_detection.constants import folder_path, stage_1_studies_count

max_slices_per_study = 64
input_slice_size = 256
batch_size = 1 # Unfortunately 64 slices is already very memory consumming, so we will keep batch_size at 1
steps_per_epoch=4
epochs = stage_1_studies_count // (steps_per_epoch * batch_size)

model_weights_path = folder_path + "/weights/classifier_3D.hdf5"

learning_rate = 1e-5
