import cv2
import numpy as np

from intracranial_hemorrhage_detection.classification_2D.params import dtype, input_image_size
from intracranial_hemorrhage_detection.preprocess import get_dicom_data

def get_input_images(filepaths):
    """
    Inputs a list of the filepaths to the dicom files and the size to which images must be resized\n
    Returns the images as resized numpy arrays. Output shape: (len(filepaths), input_image_size, input_image_size, 1)
    """
    output = []
    for filepath in filepaths:
        dicom_data = get_dicom_data(filepath)
        image = np.array(dicom_data.pixel_array, dtype=np.float32) # must use float32 for cv2.resize, even if we use another dtype later
        image = cv2.resize(image, (input_image_size, input_image_size))
        image -= np.min(image)
        image /= max(np.max(image), 1e-7)
        image = np.stack([image, image, image], axis=-1) # stacks the channel input to make rgb images
        output.append(image)
    output = np.array(output, dtype=dtype)

    min_pixel = np.min(output)
    max_pixel = np.max(output)
    assert min_pixel == 0., "Expected image pixels to be within [0, 1], but minimum is %s for %s" % (min_pixel, filepaths)
    assert max_pixel == 1., "Expected image pixels to be within [0, 1], but maximum is %s for %s" % (max_pixel, filepaths)
    assert output.shape == (len(filepaths), input_image_size, input_image_size, 3)

    return output
