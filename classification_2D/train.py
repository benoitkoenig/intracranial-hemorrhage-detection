from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import steps_per_epoch, epochs, model_weights_path, learning_rate
from intracranial_hemorrhage_detection.classification_2D.training_generator import training_generator

def train_classifier():
    callbacks = [
        ModelCheckpoint(model_weights_path),
        CSVLogger("%s/outputs/classifier_2D_training_logs.csv" % folder_path),
    ]
    gen = training_generator()
    model = get_model()
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
    model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
