# Evolving Diagnostics: Incremental and Transfer Learning in Histopathology Image Analysis for Cancer Detection

## Introduction
This project addresses the critical challenge in medical diagnostics of utilizing histopathology images for accurate cancer detection. By leveraging advanced machine learning techniques, specifically transfer and incremental learning, the project aims to enhance the precision, efficiency, and adaptability of diagnostic tools, transforming how histopathology images are interpreted and used in clinical settings.

## Objectives
1. **Transfer Learning**: Develop a robust transfer learning model using the EfficientNet architecture, trained on images from four different tumor types to serve as a versatile feature extractor for medical image analysis.
2. **Incremental Learning**: Implement incremental learning by initially training the model with 50% of the dataset, followed by five phases of incremental training, each utilizing an additional 10% of the data.

## Technologies Used
- **Python**: Primary programming language.
- **PyTorch**: Used for developing deep learning models.
- **EfficientNet**: Chosen for its efficiency and effectiveness in handling complex image data.
- **Other Tools**: NumPy, SciPy, scikit-learn for data manipulation and machine learning tasks.

## Dataset
### Training and Validation Datasets
- **Breast Cancer (BACH Dataset)**: Histopathological images representing various stages of breast cancer.
- **Prostate Cancer (SICAPv2 Dataset)**: Contains annotated prostate histology images with detailed Gleason scores.
- **Bone Cancer (UT-Osteosarcoma Dataset)**: Collection of osteosarcoma histology images.
- **Colon Cancer (CRC100K Dataset)**: Features 100,000 histological images of colorectal cancer and healthy tissues.

### Validation-Only Datasets
- **Breast Cancer (BreakHis Dataset)**: Microscopic images of breast tumor tissues, labeled as benign or malignant.
- **Cervical Cancer (SipacMed Dataset)**: Used to validate the transferability of the model.

## Model Details
The **EfficientNet B0** architecture was adapted for this project. This model was initially trained on 50% of the dataset to establish a baseline for performance. Subsequent training phases incorporated incremental learning, focusing on the retraining of the top three layers of the architecture, which allowed the model to assimilate new data while preserving previously learned knowledge.



## Detailed Results

We achieved favorable outcomes from the feature extractor training and the incremental analysis. These results show a clear improvement in model performance, validating the effectiveness of the applied machine-learning techniques.

Please refer to the project report for a detailed exploration of the results, including comprehensive comparisons and statistical validations.
[Link to Project Report](https://github.com/Anksss3d/Transfer-and-Incremental-Learning-for-Hitsopathology/blob/main/Project%20Report.pdf).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Code Description
---
## CustomDataset.py

**Description**: This Python script defines the `ImageDataset` class, a custom dataset handler for machine learning models, specifically tailored for image data. It utilizes PyTorch’s `Dataset` class to provide a framework for loading and transforming image data from a directory based on a CSV file listing. The script is designed to support both training and validation datasets with dynamic data augmentation and normalization to suit different model training scenarios.

**Key Features**:
- **Dynamic Transforms**: Implements data augmentation for training datasets to enhance model generalizability and uses simpler transformations for validation data to preserve original characteristics.
- **Class Weight Calculation**: Automatically calculates class weights to handle class imbalance during training, which is crucial for achieving fairness in model predictions across various classes.
- **Flexible Data Handling**: Supports filtering the dataset based on specific conditions to allow for more targeted analysis and training sessions.
- **Normalization**: Includes two normalization methods, one for general image data (ImageNet standards) and another specifically tuned for medical image datasets (`medned_normalizer`).

**Functionality**:
- The class initializes with paths to the data and its configuration, setting up data transformations based on the dataset type (training or validation).
- It provides methods to calculate class weights, get the length of the dataset, retrieve specific data points (images and labels), and print details about the class distribution within the dataset.
- The script enhances readability and usability in machine learning workflows by handling common preprocessing tasks within the custom dataset class.

**Usage**:
This script is used to create dataset objects that can be directly used with PyTorch data loaders for feeding into neural network models for training and validation. It simplifies the process of dataset setup, especially when dealing with complex image data and multiple classes.

---

