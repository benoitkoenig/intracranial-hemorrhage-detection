from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, save_model
from keras.optimizers import Adam
import sys

from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import steps_per_epoch, epochs, lr_init, lr_factor, lr_patience
from intracranial_hemorrhage_detection.classification_2D.training_generator import training_generator
from intracranial_hemorrhage_detection.constants import folder_path

def train_classifier():
    callbacks = [
        ModelCheckpoint(folder_path + "/weights/classifier.hdf5"),
        ReduceLROnPlateau(monitor="loss", factor=lr_factor, patience=lr_patience, verbose=1),
    ]
    gen = training_generator()
    model = get_model()
    model.compile(optimizer=Adam(lr=lr_init), loss="binary_crossentropy")
    model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
