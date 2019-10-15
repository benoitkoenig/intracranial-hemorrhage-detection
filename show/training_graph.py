import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intracranial_hemorrhage_detection.constants import training_logs_file

pd.plotting.register_matplotlib_converters()

def show_training_graph():
    df = pd.read_csv(training_logs_file)
    loss_per_epoch = df["loss"].values
    val_loss_per_epoch = df["val_loss"].values

    plt.figure(figsize=(18, 12))
    plt.plot(loss_per_epoch, color="orange")
    plt.plot(val_loss_per_epoch)
    plt.gcf().canvas.set_window_title("Loss per epoch")
    plt.title("Loss per epoch")
    plt.gca().axhline(y=.3, color="black", lw=1., alpha=.2)

    plt.show()

show_training_graph()
