import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from intracranial_hemorrhage_detection.constants import folder_path

pd.plotting.register_matplotlib_converters()

def show_loss_graph(n_dim):
    df = pd.read_csv("%s/outputs/classifier_%s_training_logs.csv" % (folder_path, n_dim))
    loss_per_epoch = df["loss"].values
    smooth_loss_per_epoch = [np.mean(loss_per_epoch[max(0, i - 20) : min(len(loss_per_epoch), i + 20)]) for i in range(len(loss_per_epoch))]

    plt.figure(figsize=(18, 12))
    plt.plot(loss_per_epoch, color="orange")
    plt.plot(smooth_loss_per_epoch)
    plt.gcf().canvas.set_window_title("Loss per epoch")
    plt.title("Loss per epoch")
    plt.gca().axhline(y=.35, color="black", lw=1., alpha=.2)

    plt.show()

# Read arguments from python command
n_dim = None
for param in sys.argv:
    if param in ["2D", "3D"]:
        n_dim = param

if (n_dim == None):
    print("Usage: python show.loss_graph.py [n_dim]")
    print("Use n_dim=2D to display the loss of the 2D classifier, n_dim=3D to display the loss of the 3D classifier")
    exit(-1)

show_loss_graph(n_dim)
