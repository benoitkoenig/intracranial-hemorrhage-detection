from intracranial_hemorrhage_detection.constants import stage_1_training_any_count

steps_per_epoch = 8
batch_size = 8
subset_size = 2 * stage_1_training_any_count
epochs = subset_size // (steps_per_epoch * batch_size)

lr_init = 2e-6
lr_factor = .2
lr_patience = 8
