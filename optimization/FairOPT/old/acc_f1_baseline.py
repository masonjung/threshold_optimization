import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc


def performance_by_threshold (y_pred_proba, y_true, threshold):
    # Compute the accuracy
    y_predicted = (y_pred_proba > threshold).astype(int)
    accuracy = accuracy_score(y_predicted, y_true)
    f1 = f1_score(y_true, y_predicted)
    
    # Percentage of data below and above the threshold
    data_below_threshold = (y_pred_proba <= threshold).mean() * 100
    data_above_threshold = (y_pred_proba > threshold).mean() * 100
    
    return accuracy, f1, data_below_threshold, data_above_threshold

def evaluate_thresholds(y_true, y_pred_proba):
    #thresholds = np.arange(0.0, 1.01, 0.01)
    results = []

    #for threshold in thresholds:
    #    accuracy, f1, data_below_threshold, data_above_threshold = performance_by_threshold (y_pred_proba, y_true, threshold)
    #    results.append({
    #        'threshold': threshold,
    #        'accuracy': accuracy,
    #        'f1': f1,
    #        'perc_below_threshold': data_below_threshold,
    #        'perc_above_threshold': data_above_threshold
    #    })

    # Static threshold
    static_threshold = 0.5
    static_accuracy, static_f1, static_data_below_threshold, static_data_above_threshold = performance_by_threshold (y_pred_proba, y_true, static_threshold)
    results.append({
        'threshold': static_threshold,
        'accuracy': static_accuracy,
        'f1': static_f1,
        'perc_below_threshold': static_data_below_threshold,
        'perc_above_threshold': static_data_above_threshold
    })

    # ROC-based threshold
    """
    ROC curve is a graphical representation used to evaluate the performance of a binary classification model.
    It shows the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR)
        at various threshold settings.
    ROC curve visualizes the model's trade-offs between TPR and FPR at different thresholds.
    AUC summarizes the quality of the ROC curve into a single number (closer to 1 is better).
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_best_idx = np.argmax(tpr - fpr)
    roc_threshold = roc_thresholds[roc_best_idx]

    roc_accuracy, roc_f1, roc_data_below_threshold, roc_data_above_threshold = performance_by_threshold (y_pred_proba, y_true, roc_threshold)
    results.append({
        'threshold': roc_threshold,
        'accuracy': roc_accuracy,
        'f1': roc_f1,
        'perc_below_threshold': roc_data_below_threshold,
        'perc_above_threshold': roc_data_above_threshold
    })

    # Convert the results List to a DataFrame:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='threshold', ascending=True)
    results_df = results_df.drop_duplicates(subset='threshold', keep='first')
    results_df = results_df.reset_index(drop=True)
    
    return results_df, static_threshold, roc_threshold