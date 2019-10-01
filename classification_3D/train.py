from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.optimizers import Adam

from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.classification_3D.model import get_model
from intracranial_hemorrhage_detection.classification_3D.params import epochs, model_weights_path, steps_per_epoch, learning_rate
from intracranial_hemorrhage_detection.classification_3D.training_generator import training_generator

def train_classifier():
    callbacks = [
        ModelCheckpoint(model_weights_path),
        CSVLogger("%s/outputs/classifier_3D_training_logs.csv" % folder_path),
    ]
    model = get_model()
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="binary_crossentropy")
    model.fit_generator(training_generator(), steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

train_classifier()
