import cv2
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.preprocess import preprocess_voxels
from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.preprocess import get_dicom_data

def check_dicom_data_consistency(dicom_datas):
    "Given the dicom_data for a specific study, check the metadata consistency"
    dicom_data_0 = dicom_datas[0]
    assertion_error_message = "Assertion error for instance id %s" % dicom_data_0["0020", "000d"][0:]
    for dicom_data in dicom_datas:
        assert dicom_data["0020", "0032"][:2] == dicom_data_0["0020", "0032"][:2], assertion_error_message
        assert dicom_data["0020", "000d"] == dicom_data_0["0020", "000d"], assertion_error_message
        assert dicom_data["0020", "0037"] == dicom_data_0["0020", "0037"], assertion_error_message
        assert dicom_data["0028", "0030"] == dicom_data_0["0028", "0030"], assertion_error_message
        assert dicom_data["0028", "0010"] == dicom_data_0["0028", "0010"], assertion_error_message
        assert dicom_data["0028", "0011"] == dicom_data_0["0028", "0011"], assertion_error_message
        assert dicom_data["0028", "1050"] == dicom_data_0["0028", "1050"], assertion_error_message
        assert dicom_data["0028", "1051"] == dicom_data_0["0028", "1051"], assertion_error_message
        assert dicom_data["0028", "1052"] == dicom_data_0["0028", "1052"], assertion_error_message
        assert dicom_data.RescaleSlope == 1, assertion_error_message

def generate_3D_image(slices_ids, dtype, input_slice_size, folder="stage_1_train"):
    "All input slice ids must have the same study id. Outputs the voxels reconstructed for the given slices"
    slices_filepaths = ["%s/data/%s_images/%s.dcm" % (folder_path, folder, id) for id in slices_ids]
    unsorted_dicom_datas = [get_dicom_data(filepath) for filepath in slices_filepaths]
    check_dicom_data_consistency(unsorted_dicom_datas)
    dicom_datas = sorted(unsorted_dicom_datas, key=lambda x: x["0020", "0032"][2])
    voxels = [dicom_data.pixel_array for dicom_data in dicom_datas]
    voxels = preprocess_voxels(voxels, dtype, input_slice_size)
    return voxels
