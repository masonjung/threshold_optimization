import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# check if this is imported
print("utils.py is imported")

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for binary classification.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Predicted binary labels.

    Returns:
    dict: Dictionary of calculated metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

def confusion_matrix_metrics(y_true, y_pred):
    """
    Calculate confusion matrix and related metrics.

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Predicted binary labels.

    Returns:
    dict: Dictionary with confusion matrix and derived metrics (TP, FP, TN, FN).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }

def apply_threshold(y_scores, threshold):
    """
    Apply a threshold to predicted probabilities to generate binary predictions.

    Parameters:
    y_scores (array-like): Predicted probabilities or confidence scores.
    threshold (float): Threshold value to classify as positive.

    Returns:
    array-like: Binary predictions based on the threshold.
    """
    return (np.array(y_scores) >= threshold).astype(int)

def split_data_by_groups(y_true, y_scores, groups):
    """
    Split data into subgroups based on group labels.

    Parameters:
    y_true (array-like): True binary labels.
    y_scores (array-like): Predicted probabilities or confidence scores.
    groups (array-like): Group identifiers for each instance.

    Returns:
    dict: A dictionary with group labels as keys and tuples of (y_true, y_scores) as values.
    """
    unique_groups = np.unique(groups)
    grouped_data = {}

    for group in unique_groups:
        mask = (groups == group)
        grouped_data[group] = (np.array(y_true)[mask], np.array(y_scores)[mask])

    return grouped_data
