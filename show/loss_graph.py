import matplotlib.pyplot as plt
import pandas as pd
import sys

from intracranial_hemorrhage_detection.constants import folder_path

pd.plotting.register_matplotlib_converters()

def get_smooth_loss_per_epoch(loss_per_epoch):
    value = loss_per_epoch[0]
    smooth_loss_per_epoch = []
    momentum = .9
    for i in loss_per_epoch:
        value = momentum * value + (1 - momentum) * i
        smooth_loss_per_epoch.append(value)
    return smooth_loss_per_epoch

def show_loss_graph(n_dim):
    df = pd.read_csv("%s/outputs/classifier_%sD_training_logs.csv" % (folder_path, n_dim))
    loss_per_epoch = df["loss"].values

    plt.figure(figsize=(18, 12))
    plt.plot(loss_per_epoch, color="orange")
    plt.plot(get_smooth_loss_per_epoch(loss_per_epoch))
    plt.gcf().canvas.set_window_title("Loss per epoch")
    plt.title("Loss per epoch")
    plt.gca().axhline(y=.35, color="black", lw=1., alpha=.2)

    plt.show()

# Read arguments from python command
n_dim = None
for param in sys.argv:
    if param in ["2", "3"]:
        n_dim = param

if (n_dim == None):
    print("Usage: python show.loss_graph.py [n_dim]")
    print("Use n_dim=2 to display the loss of the 2D classifier, n_dim=3 to display the loss of the 3D classifier")
    exit(-1)

show_loss_graph(n_dim)
