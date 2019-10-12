import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from intracranial_hemorrhage_detection.constants import training_logs_file

pd.plotting.register_matplotlib_converters()

def show_training_graph():
    df = pd.read_csv(training_logs_file)
    loss_per_epoch = df["loss"].values
    smooth_loss_per_epoch = [np.mean(loss_per_epoch[max(0, i - 20) : min(len(loss_per_epoch), i + 20)]) for i in range(len(loss_per_epoch))]

    plt.figure(figsize=(18, 12))
    plt.plot(loss_per_epoch, color="orange")
    plt.plot(smooth_loss_per_epoch)
    plt.gcf().canvas.set_window_title("Loss per epoch")
    plt.title("Loss per epoch")
    plt.gca().axhline(y=.35, color="black", lw=1., alpha=.2)

    plt.show()

show_training_graph()
