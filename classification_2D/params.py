import numpy as np

from intracranial_hemorrhage_detection.constants import folder_path, stage_1_training_any_count

steps_per_epoch = 80
batch_size = 8
half_test_size = 463 # this half_test_size allows epochs to be naturally round
half_train_size = stage_1_training_any_count - half_test_size
real_epochs = 20

assert (2 * half_train_size) % (steps_per_epoch * batch_size) == 0
epochs = real_epochs * 2 * half_train_size // (steps_per_epoch * batch_size)

input_image_size = 224
learning_rate = 5e-5
model_weights_path = folder_path + "/weights/classifier.hdf5"
dtype = np.float32
