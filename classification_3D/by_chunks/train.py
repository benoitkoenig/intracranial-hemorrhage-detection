from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.constants import training_logs_file
from intracranial_hemorrhage_detection.classification_3D.by_chunks.model import get_model
from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import epochs, model_weights_path, steps_per_epoch, learning_rate
from intracranial_hemorrhage_detection.classification_3D.by_chunks.training_generator import training_generator

def train_classifier():
    callbacks = [
        ModelCheckpoint(model_weights_path),
        CSVLogger(training_logs_file),
    ]
    model = get_model()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy")
    model.fit_generator(training_generator(), steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
