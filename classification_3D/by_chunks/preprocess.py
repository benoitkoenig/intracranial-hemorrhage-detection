import math

from intracranial_hemorrhage_detection.classification_3D.by_chunks.params import chunk_size
from intracranial_hemorrhage_detection.classification_3D.preprocess import get_study_wise_data

def get_chunk_wise_data(folder):
    "Load the study-wise data and splits it in chunks of voxels. Folder should be one of 'stage_1_train', 'stage_1_test'"
    assert (folder in ["stage_1_train", "stage_1_test"])

    all_chunks = []
    study_wise_images = get_study_wise_data(folder)
    for image_ids in study_wise_images:
        nb_chunks = math.ceil(len(image_ids) / chunk_size)
        for i in range(nb_chunks - 1):
            all_chunks.append(image_ids[i * nb_chunks : i * nb_chunks + chunk_size])
        all_chunks.append(image_ids[-chunk_size:])
    return all_chunks
