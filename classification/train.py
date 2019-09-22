from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model, save_model
from keras.optimizers import Adam
import sys
import tensorflow as tf

from intracranial_hemorrhage_detection.classification.get_classifier import get_classifier
from intracranial_hemorrhage_detection.classification.params import steps_per_epoch, epochs, lr_init, lr_factor, lr_patience, lr_min
from intracranial_hemorrhage_detection.classification.training_generator import training_generator
from intracranial_hemorrhage_detection.constants import folder_path

def train_classifier(backbone_name):
    graph = tf.Graph()
    callbacks = [
        ModelCheckpoint(folder_path + "/weights/hydra_%s_body.hdf5" % backbone_name),
        ReduceLROnPlateau(monitor="loss", factor=lr_factor, patience=lr_patience, min_lr=lr_min),
    ]
    gen = training_generator(graph)
    with graph.as_default():
        model = get_classifier(backbone_name)
        model.compile(optimizer=Adam(lr=lr_init), loss="binary_crossentropy")
        model.fit_generator(gen, steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks)

# Read arguments from python command
backbone_name = None
for param in sys.argv:
    if param in ["resnet50", "densenet169"]:
        backbone_name = param

if (backbone_name == None):
    print("Usage: python classification.train.py [resnet50|densenet169]")
    exit(-1)

train_classifier(backbone_name)
