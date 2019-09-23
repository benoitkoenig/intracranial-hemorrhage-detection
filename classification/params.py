from intracranial_hemorrhage_detection.constants import stage_1_training_count

steps_per_epoch = 15
epochs = stage_1_training_count // steps_per_epoch

lr_init = 1e-4
lr_factor = .8
lr_patience = 5
lr_min = 1e-6
