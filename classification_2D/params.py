from intracranial_hemorrhage_detection.constants import folder_path, stage_1_training_any_count

steps_per_epoch = 8
batch_size = 8
subset_size = 2 * stage_1_training_any_count
epochs = subset_size // (steps_per_epoch * batch_size)

input_image_size = 256
model_weights_path = folder_path + "/weights/classifier.hdf5"
learning_rate = 1e-5
