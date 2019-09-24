from intracranial_hemorrhage_detection.constants import stage_1_training_any_count

steps_per_epoch = 15
subset_size = 2 * stage_1_training_any_count
epochs = subset_size // steps_per_epoch

lr_init = 1e-4
lr_factor = .2
lr_patience = 8
