## Intracranial Hemorrhage Detection

This repository is my solution to this [Kaggle competition](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

Author: Benoît Koenig

### Usage:

- Train the U-Net classifier with a specific backbone: python -m classification.train [resnet50|densenet169]

- Visualize a single dicom file: python -m show.dicom_file [stage_1_train|stage_1_test] [index]

#### Note

tensorflow-gpu is not included in requirements.txt as it is not always relevant. To use the GPU, install tensorflow-gpu via "pip install tensorflow-gpu"
