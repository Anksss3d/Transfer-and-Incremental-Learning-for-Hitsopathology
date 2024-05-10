
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve
from imblearn.metrics import specificity_score
import platform
import torch
import matplotlib.pyplot as plt


MODEL_SAVE_DIR = 'saved_models/'
BASE_MODEL_NAME = "transfer_efficientB0_base_14class.pt"

generic_dir = ""

os_name = platform.system()
if os_name == 'Windows':
    generic_dir = r"D://datasets/all-multi-cancer/"
elif os_name == 'Darwin':
    generic_dir = r"/Users/anksss3d/datasets/multi-cancer/"
else:
    generic_dir = None

print("Generic Dir", generic_dir)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def adjust_model_trainability(model, num_blocks_to_unfreeze):
    """
    Adjust the trainability of the model by freezing or unfreezing its parameters.

    :param model: The model to adjust trainability of.
    :param num_blocks_to_unfreeze: The number of blocks to unfreeze from the model.

    :return: The model with adjusted trainability.
    """
    # Index array indicating where each block starts
    block_start_indices = [0, 1, 3, 5, 8, 12, 15]

    # First, freeze all parameters in the model
    for param in model.parameters():
        param.requires_grad = False

    # Calculate the starting index for unfreezing
    # -1 because the block_start_indices includes 0 as the first element for the first block
    start_unfreeze_index = len(block_start_indices) - num_blocks_to_unfreeze - 1

    # Unfreeze the parameters from the determined start block to the end
    for i in range(block_start_indices[start_unfreeze_index], len(model._blocks)):
        for param in model._blocks[i].parameters():
            param.requires_grad = True

    return model


def hh_mm_ss(seconds):
    """
    This method converts the given number of seconds into a string representing the time in hours, minutes, and seconds.

    :param seconds: The number of seconds to convert into time.
    :return: A string representing the time in the format "hh:mm:ss".
    """
    seconds = int(seconds)
    hrs, mins, secs = seconds // 3600, (seconds % 3600) // 60, (seconds % 3600) % 60
    return  f"{hrs:02d}:{mins:02d}:{secs:02d}"



def calculate_eta(completed, total, total_time_taken_till_now):
    """
    Calculates the estimated time of arrival (ETA) based on the number of tasks completed,
    total number of tasks, and the total time taken till now.

    :param completed: The number of tasks completed.
    :param total: The total number of tasks.
    :param total_time_taken_till_now: The total time taken till now.
    :return: The estimated time of arrival (ETA) in the format "hh:mm:ss".

    Example usage:
    >>> calculate_eta(10, 100, 1800)
    '03:00:00'
    """
    if completed == 0:
        eta = "N/A"
    else:
        time_per_task = total_time_taken_till_now / completed
        remaining_tasks = total - completed
        eta = int(time_per_task * remaining_tasks)
        eta = hh_mm_ss(eta)
    return eta



def calculate_metrics(y_true, y_pred, y_prob):
    """
    Calculate metrics for multi-class classification.

    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :param y_prob: Predicted probabilities for each class.
    :return: DataFrame containing metrics for each class and confusion matrix.

    """
    # Get the number of classes
    num_classes = len(np.unique(y_true))
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if isinstance(y_prob, list):
        y_prob = np.array(y_prob)

    # Initialize dictionaries to store metrics for each class
    metrics_dict = {
        'Class': list(range(num_classes)),
        'Specificity': [],
        'Sensitivity': [],
        'F1 Score': [],
        'AUC': []
    }

    # Calculate metrics for each class
    for class_label in range(num_classes):
        # Create binary vectors for the current class
        # print(y_true)
        y_true_class = (y_true == class_label).astype(int)
        y_pred_class = (y_pred == class_label).astype(int)


        # Calculate confusion matrix
        cm = confusion_matrix(y_true_class, y_pred_class)

        # Calculate specificity
        specificity = specificity_score(y_true_class, y_pred_class)

        # Calculate sensitivity (Recall)
        recall = np.diag(cm) / np.sum(cm, axis=1)

        # Calculate precision
        precision = np.diag(cm) / np.sum(cm, axis=0)

        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Calculate AUC
        precision, recall, _ = precision_recall_curve(y_true_class, y_prob[:, class_label])
        auc_score = auc(recall, precision)

        # Append metrics to the dictionary
        metrics_dict['Specificity'].append(specificity)
        metrics_dict['Sensitivity'].append(recall[0])
        metrics_dict['F1 Score'].append(f1_score[0])
        metrics_dict['AUC'].append(auc_score)

    # Create a DataFrame from the metrics dictionary
    metrics_df = pd.DataFrame(metrics_dict)

    return metrics_df, confusion_matrix(y_true, y_pred)


def calculate_metrics2(y_true, y_pred, y_pred_proba, plot_path=None):
    """
    Calculate metrics including Specificity, Sensitivity, F1 Score, AUC, Confusion Matrix, and plot the ROC curve.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param y_pred_proba: The predicted probabilities.
    :param plot_path: The file path to save the ROC plot. (optional)
    :return: None
    """
    # Calculate Specificity and Sensitivity
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    # Calculate F1 Score
    f1_score = 2 * (sensitivity * (1 - specificity)) / (sensitivity + (1 - specificity))

    # Calculate AUC (Area Under the ROC Curve)
    if y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
    auc = roc_auc_score(y_true, y_pred_proba)

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Create a dictionary to store the results
    metrics_dict = {
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "F1 Score": f1_score,
        "AUC": auc,
        "Confusion Matrix": conf_matrix
    }

    # Calculate ROC curve and plot it
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")

    # Save the plot to the specified path
    if plot_path:
        plt.savefig(plot_path)

    plt.close()

    return None