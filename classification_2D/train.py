from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.constants import training_logs_file
from intracranial_hemorrhage_detection.classification_2D.model import get_model
from intracranial_hemorrhage_detection.classification_2D.params import epochs, learning_rate, model_weights_path, steps_per_epoch
from intracranial_hemorrhage_detection.classification_2D.training_generator import training_generator

def train_classifier():
    callbacks = [
        ModelCheckpoint(model_weights_path, save_weights_only=True),
        CSVLogger(training_logs_file),
    ]

    gen = training_generator()
    validation_data = next(gen)

    model = get_model()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy")
    model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=validation_data, callbacks=callbacks)

train_classifier()
