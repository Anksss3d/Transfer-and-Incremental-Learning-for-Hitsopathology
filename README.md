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


## utilities.py

**Description**: This Python script encapsulates a suite of utility functions designed to support machine learning models in evaluating performance and managing training dynamics. It includes functionality for adjusting model trainability, calculating various performance metrics, estimating completion time for training tasks, and dynamically handling directory paths based on the operating system.

**Key Features**:
- **Dynamic Directory Management**: Automatically adjusts directory paths for data and models based on the operating system, ensuring compatibility across Windows, macOS, and Unix-like systems.
- **Model Trainability Adjustment**: Provides a function to selectively freeze or unfreeze layers of a neural network, facilitating fine-tuning and transfer learning processes.
- **Performance Metrics Calculation**: Offers comprehensive functions to calculate specificity, sensitivity, F1 score, AUC, and produce confusion matrices for classification tasks. It also supports generating ROC curves and plotting them if a path is provided.
- **Time Management Utilities**: Includes functions to estimate the remaining time for tasks (ETA) and convert seconds to a more readable format (hh:mm:ss), aiding in monitoring and managing long-running training processes.

**Functionality**:
- The `adjust_model_trainability` function enables precise control over which layers of a model are trainable, aiding in effective fine-tuning.
- `calculate_metrics` and `calculate_metrics2` provide detailed performance analysis for classification outcomes, which is crucial for evaluating and improving model accuracy.
- Time-related utilities like `hh_mm_ss` and `calculate_eta` help track and predict training durations, improving usability during model development and training sessions.

**Usage**:
This script is intended to be used as a helper module in machine learning projects where managing model training phases, evaluating model performance, and ensuring cross-platform compatibility are required. It simplifies routine tasks and enhances the maintainability of the codebase by centralizing common operations.

---

## train_model.py

**Description**: This Python script is the main driver for training machine learning models, specifically focusing on EfficientNet architectures. It integrates several custom utilities and dataset handling functionalities to streamline the training and validation processes for image classification tasks.

**Key Features**:
- **Model Training and Validation**: Implements a comprehensive training loop that includes forward passes, loss calculation, backpropagation, and parameter updates. It also handles validation using a separate dataset to monitor model performance and generalize ability.
- **Metrics Tracking and Display**: Utilizes a custom method to print detailed training metrics including loss, epoch progress, and estimated time of arrival (ETA) for remaining training.
- **Dynamic Model Adjustments**: Employs functions from `utilities.py` to adjust model trainability by selectively unfreezing model layers, which is critical for fine-tuning pre-trained models.
- **Performance Evaluation**: After each training epoch, the script evaluates the model on the validation set, calculating standard classification metrics and a confusion matrix to assess model accuracy and other performance metrics.

**Functionality**:
- The `get_datasets` function initializes training and validation datasets using configurations specified in a dictionary, facilitating easy switches between different sets of data.
- `run_epoch` manages the operations within a single epoch, alternating between training and validation phases, and invokes metric calculation and logging.
- The `train` function sets up the model, datasets, loss function, and optimizer based on provided configurations. It iterates over a specified number of epochs, saving the model after each epoch and logging detailed performance data.

**Usage**:
This script is typically executed to train a new model from scratch or continue training from a pre-trained state. It requires a configuration dictionary that specifies paths, hyperparameters, and model details. This script is critical for automating the training process, allowing for extensive experimentation and iterative improvements in model performance.

---
