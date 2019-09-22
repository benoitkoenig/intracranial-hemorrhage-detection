import matplotlib.pyplot as plt
import sys

from intracranial_hemorrhage_detection.constants import labels
from intracranial_hemorrhage_detection.preprocess import get_all_images_list, get_dicom_data, get_all_true_labels

def show_data(folder, index):
    images_list = get_all_images_list(folder)

    # Check index is valid
    if index >= len(images_list):
        print("Index %s out of range. Max index is %s" % (index, len(images_list) - 1))
        exit(-1)

    (id, filepath) = images_list[index]
    dicom_data = get_dicom_data(filepath)

    print("\n\nImage ID: %s\n" % id)
    print(dicom_data)
    if (folder == "stage_1_train"):
        all_true_labels = get_all_true_labels()
        true_labels = all_true_labels[id]
        print("\nReal labels are:")
        for i in range(len(labels)):
            print("%s: %s" % (labels[i], true_labels[i] == 1))

    # Display the data and image through matplotlib
    plt.imshow(dicom_data.pixel_array)
    plt.show()

# Read arguments from python command
folder = None
index = None
for param in sys.argv:
    if param in ["stage_1_train", "stage_1_test"]:
        folder = param
    if param.isdigit():
        index = int(param)

if (index == None):
    print("Usage: python show.dicom_file.py [stage_1_train|stage_1_test] [index]")
    exit(-1)
if (folder == None):
    print("No folder specified. Defaults to stage_1_train")
    folder = "stage_1_train"

show_data(folder, index)
