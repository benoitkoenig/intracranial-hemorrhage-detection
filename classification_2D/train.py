from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import epochs, model_weights_path, steps_per_epoch
from intracranial_hemorrhage_detection.classification_2D.training_generator import training_generator

def train_classifier():
    callbacks = [
        ModelCheckpoint(model_weights_path),
        CSVLogger("%s/outputs/classifier_2D_training_logs.csv" % folder_path),
        LearningRateScheduler(lambda epoch, _: 1e-5 if (epoch < 60) else 1e-6),
    ]
    model = get_model()
    model.compile(optimizer=Adam(), loss="binary_crossentropy")
    model.fit_generator(training_generator(), steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
