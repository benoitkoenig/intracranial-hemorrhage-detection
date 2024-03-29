import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from intracranial_hemorrhage_detection.classification_3D.generate_3D_image import generate_3D_image
from intracranial_hemorrhage_detection.constants import folder_path

def show_reconstruction_3D(study_id):
    df = pd.read_csv("%s/outputs/study_ids_stage_1_train.csv" % folder_path)
    slice_ids = eval(df[df["study_id"] == study_id]["slice_ids"].values[0])
    voxels = generate_3D_image(slice_ids, dtype=np.float32, input_slice_size=512)
    voxels = np.reshape(voxels, voxels.shape[:3]) # Remove the channels, which is a singleton

    plt.figure(figsize=(18, 12))
    fig = plt.gcf()
    fig.canvas.set_window_title("3D reconstruction of %s" % study_id)
    plt.title("3D reconstruction of %s" % study_id)
    animation_steps = [[plt.imshow(voxel_slice)] for voxel_slice in voxels]
    animation.ArtistAnimation(fig, animation_steps, interval=200, blit=True)
    plt.show()

# Read arguments from python command
study_id = None
for param in sys.argv:
    if param[:3] == "ID_":
        study_id = param

if (study_id == None):
    print("Usage: python show.reconstruction_3D.py [study_id]\nEx: ID_2e3ced1a90, ID_ebcf07d991, ID_b3124df7de")
    exit(-1)

show_reconstruction_3D(study_id)
