import cv2
import numpy as np

from intracranial_hemorrhage_detection.classification_3D.params import input_slice_size
from intracranial_hemorrhage_detection.classification_3D.preprocess import preprocess_voxels
from intracranial_hemorrhage_detection.constants import folder_path
from intracranial_hemorrhage_detection.preprocess import get_dicom_data

def check_dicom_data_consistency(dicom_datas):
    "Given the dicom_data for a specific study, check the metadata consistency"
    dicom_data_0 = dicom_datas[0]
    for dicom_data in dicom_datas:
        assert dicom_data["0020", "0032"][:2] == dicom_data_0["0020", "0032"][:2]
        assert dicom_data["0020", "000d"] == dicom_data_0["0020", "000d"]
        assert dicom_data["0020", "0037"] == dicom_data_0["0020", "0037"]
        assert dicom_data["0028", "0030"] == dicom_data_0["0028", "0030"]
        assert dicom_data["0028", "0010"] == dicom_data_0["0028", "0010"]
        assert dicom_data["0028", "0011"] == dicom_data_0["0028", "0011"]
        assert dicom_data["0028", "1050"] == dicom_data_0["0028", "1050"]
        assert dicom_data["0028", "1051"] == dicom_data_0["0028", "1051"]
        assert dicom_data["0028", "1052"] == dicom_data_0["0028", "1052"]
        assert dicom_data.RescaleSlope == 1

def generate_3D_image(slices_ids):
    "All input slice ids must have the same study id. Outputs the voxels reconstructed for the given slices"
    slices_filepaths = ["%s/data/stage_1_train_images/%s.dcm" % (folder_path, id) for id in slices_ids]
    unsorted_dicom_datas = [get_dicom_data(filepath) for filepath in slices_filepaths]
    check_dicom_data_consistency(unsorted_dicom_datas)
    dicom_datas = sorted(unsorted_dicom_datas, key=lambda x: x["0020", "0032"][2])
    voxels = [dicom_data.pixel_array for dicom_data in dicom_datas]
    voxels = preprocess_voxels(voxels)
    return voxels
