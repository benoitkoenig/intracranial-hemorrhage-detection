from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import steps_per_epoch, epochs, learning_rate
from intracranial_hemorrhage_detection.classification_2D.training_generator import training_generator
from intracranial_hemorrhage_detection.constants import folder_path

def train_classifier():
    callbacks = [
        ModelCheckpoint(folder_path + "/weights/classifier.hdf5"),
    ]
    gen = training_generator()
    model = get_model()
    model.compile(optimizer=Adam(lr=learning_rate), loss="binary_crossentropy")
    model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
