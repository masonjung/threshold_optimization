from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
import pandas as pd

def calculate_optimal_threshold(file_path, true_label_column, pred_columns, fpr_limit=0.1): # change to 0.01 to get FPR < 0.01
    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract true labels and predicted probabilities
    true_labels = df[true_label_column]
    avg_pred = df[pred_columns].mean(axis=1)

    # Compute the ROC curve and AUROC
    fpr, tpr, thresholds = roc_curve(true_labels, avg_pred)
    auroc = roc_auc_score(true_labels, avg_pred)

    # Filter thresholds for FPR < fpr_limit
    valid_indices = np.where(fpr < fpr_limit)[0]

    # Identify the optimal threshold: highest TPR under FPR constraint
    optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]
    optimal_threshold = thresholds[optimal_idx]

    # Compile results
    results = {
        "Optimal Threshold": optimal_threshold,
        "AUROC": auroc,
        "FPR at Optimal Threshold": fpr[optimal_idx],
        "TPR at Optimal Threshold": tpr[optimal_idx]
    }

    return results


# Run the optimization method 2
file_path = [PATH]
true_label_column = 'AI_written'
pred_columns = [
    # 'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    # 'radar_probability'
]

results = calculate_optimal_threshold(file_path, true_label_column, pred_columns)
print(results)

AUROC_threshold = results["Optimal Threshold"]
print(AUROC_threshold)

import matplotlib.pyplot as plt



def plot_roc_curve_with_threshold(file_path, true_label_column, pred_columns, fpr_limit=0.1):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract true labels and predicted probabilities
    true_labels = df[true_label_column]
    avg_pred = df[pred_columns].mean(axis=1)

    # Compute the ROC curve and AUROC
    fpr, tpr, thresholds = roc_curve(true_labels, avg_pred)
    auroc = roc_auc_score(true_labels, avg_pred)
    
    # Filter thresholds for FPR < fpr_limit
    valid_indices = np.where(fpr < fpr_limit)[0]
    optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]
    optimal_threshold = thresholds[optimal_idx]

    # Visualize the ROC curve and highlight the region where FPR < fpr_limit
    plt.figure(figsize=(10, 6))

    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f"ROC Curve (AUROC = {auroc:.4f})", linewidth=2)

    # Highlight the region where FPR < fpr_limit
    plt.fill_between(fpr, tpr, where=(fpr < fpr_limit), color='green', alpha=0.4) # , label=f"FPR < {fpr_limit:.7f}"

    # Mark the optimal threshold point
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='green', label=f"Optimal Threshold = {optimal_threshold:.7f}", zorder=7)

    # mark the FPR 10% line
    plt.axvline(x=fpr_limit, color='red', linestyle='--', label=f"FPR = {fpr_limit:.2f}")

    # Add labels and legend
    plt.title("Optimizing Threshold using AUROC under the FPR constraint")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.1)


    # Set axis limits and ticks
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))


    # Show the plot
    plt.show()

# Run the visualization function
plot_roc_curve_with_threshold(file_path, true_label_column, pred_columns)
