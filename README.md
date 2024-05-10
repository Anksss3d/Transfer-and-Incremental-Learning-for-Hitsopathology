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

For a detailed exploration of the results, including comprehensive comparisons and statistical validations, please refer to the project report [Link to Project Report]([https://github.com/username/repository/blob/main/Project%20Report.pdf](https://github.com/Anksss3d/Transfer-and-Incremental-Learning-for-Hitsopathology/blob/main/Project%20Report.pdf)) .
