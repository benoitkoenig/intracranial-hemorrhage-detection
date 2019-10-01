## Intracranial Hemorrhage Detection

This repository is my solution to this [Kaggle competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

Author: Benoît Koenig

### Usage:

- Train the 2D classifier: python -m classification_2D.train

- Group dicom datas by study ids and store them in study_ids.csv: python -m create_study_wise_data

- Calculate and export predictions to csv: python -m export_predictions

- Visualize a single dicom file: python -m show.dicom_file [stage_1_train|stage_1_test] [index]

- Visualize the evolution of the loss for the 2D or 3D classifier: python -m show.loss_graph [2|3]

- Visualize the 3D reconstruction for the 3D classifier: python -m show.reconstruction_3D

- Move all probabilities upward or downward and save them in slided_results.csv: python -m export_predictions.slide_predictions [sliding_param]
