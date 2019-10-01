import matplotlib.pyplot as plt
import pandas as pd

from intracranial_hemorrhage_detection.constants import folder_path

pd.plotting.register_matplotlib_converters()

df = pd.read_csv("%s/outputs/classifier_2D_training_logs.csv" % folder_path)
loss_per_epoch = df["loss"].values

def get_smooth_loss_per_epoch(loss_per_epoch):
    value = loss_per_epoch[0]
    smooth_loss_per_epoch = []
    momentum = .9
    for i in loss_per_epoch:
        value = momentum * value + (1 - momentum) * i
        smooth_loss_per_epoch.append(value)
    return smooth_loss_per_epoch

plt.figure(figsize=(18, 12))
plt.plot(loss_per_epoch, color="orange")
plt.plot(get_smooth_loss_per_epoch(loss_per_epoch))
plt.gcf().canvas.set_window_title("Loss per epoch")
plt.title("Loss per epoch")

plt.show()
