from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_auroc_based_accuracy_threshold(file_path, true_label_column, pred_columns):

    df = pd.read_csv(file_path)

    # Extract true labels and predicted probabilities
    true_labels = df[true_label_column]
    avg_pred = df[pred_columns].mean(axis=1)

    # ROC curve and AUROC
    fpr, tpr, thresholds = roc_curve(true_labels, avg_pred)
    auroc = roc_auc_score(true_labels, avg_pred)

    # ACC for each threshold
    accuracies = []
    for threshold in thresholds:
        predicted_labels = (avg_pred >= threshold).astype(int)
        accuracy = accuracy_score(true_labels, predicted_labels)
        accuracies.append(accuracy)

    optimal_idx = np.argmax(accuracies)  # Find index of maximum raw accuracy
    optimal_threshold = thresholds[optimal_idx]

    # Compile results
    results = {
        "Optimal Threshold": optimal_threshold,
        "AUROC": auroc,
        "FPR at Optimal Threshold": fpr[optimal_idx],
        "TPR at Optimal Threshold": tpr[optimal_idx],
        "Thresholds": thresholds,
        "Accuracies": accuracies,
    }

    return results

# Run the optimization method
file_path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_Mage_d3.csv"
true_label_column = 'AI_written'
pred_columns = [
    'roberta_large_openai_detector_probability',
]

results = calculate_auroc_based_accuracy_threshold(file_path, true_label_column, pred_columns)
print(results)