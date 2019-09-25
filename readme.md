## Intracranial Hemorrhage Detection

This repository is my solution to this [Kaggle competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

Author: Benoît Koenig

### Usage:

- Train the 2D classifier: python -m classification_2D.train

- Visualize a single dicom file: python -m show.dicom_file [stage_1_train|stage_1_test] [index]

- Group dicom datas by study ids and store them in csv: python -m create_study_wise_data
