from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import steps_per_epoch, epochs, model_weights_path
from intracranial_hemorrhage_detection.classification_2D.training_generator import training_generator

def lr_schedule(epoch_index, _):
    if epoch_index < epochs // 5:
        return 1e-5
    if epoch_index < epochs // 2:
        return 1e-6
    return 1e-7

def train_classifier():
    callbacks = [
        ModelCheckpoint(model_weights_path),
        LearningRateScheduler(lr_schedule),
        CSVLogger("%s/outputs/classifier_2D_training_logs.csv" % folder_path),
    ]
    gen = training_generator()
    model = get_model()
    model.compile(optimizer=Adam(), loss="binary_crossentropy")
    model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
