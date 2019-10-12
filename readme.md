## Intracranial Hemorrhage Detection

This repository is my solution to this [Kaggle competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

Author: Benoît Koenig

### Usage:

- Train the 2D classifier: python -m classification_2D.train

- Train the 3D classifier by whole study: python -m classification_3D.by_whole_study.train

- Group dicom datas by study ids and store them in csv: python -m classification_3D.group_dicom_files_by_study_id [stage_1_train|stage_1_test]

- Calculate and export predictions from classifier_2D to csv: python -m export_predictions.classifier_2D

- Calculate and export predictions from classifier_3D by whole study to csv: python -m export_predictions.classifier_3D_by_whole_study

- Visualize a single dicom file: python -m show.dicom_file [stage_1_train|stage_1_test] [index]

- Visualize the evolution of the loss for the 2D or 3D classifier: python -m show.loss_graph [2D|3D]

- Visualize the 3D reconstruction for the 3D classifier: python -m show.reconstruction_3D

- Move all probabilities in results.csv upward or downward and save them in slided_results.csv: python -m export_predictions.slide_predictions [sliding_param]
