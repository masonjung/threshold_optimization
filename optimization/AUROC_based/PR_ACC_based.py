from sklearn.metrics import precision_recall_curve, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_pr_based_accuracy_threshold(file_path, true_label_column, pred_columns):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract true labels and predicted probabilities
    true_labels = df[true_label_column]
    avg_pred = df[pred_columns].mean(axis=1)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_labels, avg_pred)

    # ACC for each threshold
    accuracies = []
    for threshold in thresholds:
        predicted_labels = (avg_pred >= threshold).astype(int)
        accuracy = accuracy_score(true_labels, predicted_labels)
        accuracies.append(accuracy)

    # Optimal threshold: maximum accuracy
    optimal_idx = np.argmax(accuracies)
    optimal_threshold = thresholds[optimal_idx]

    # Compile results
    results = {
        "Optimal Threshold": optimal_threshold,
        "Precision at Optimal Threshold": precision[optimal_idx],
        "Recall at Optimal Threshold": recall[optimal_idx],
        "Max Accuracy": accuracies[optimal_idx],
        "Thresholds": thresholds,
        "Accuracies": accuracies,
        "Precision": precision[:-1],
        "Recall": recall[:-1],
    }

    return results

def plot_pr_curve_with_accuracy(results):
    # Extract metrics from results
    precision = results["Precision"]
    recall = results["Recall"]
    thresholds = results["Thresholds"]
    accuracies = results["Accuracies"]
    optimal_threshold = results["Optimal Threshold"]
    max_accuracy = results["Max Accuracy"]

    # Plot Precision-Recall curve
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label="Precision-Recall Curve", linewidth=2)
    plt.scatter(results["Recall at Optimal Threshold"], results["Precision at Optimal Threshold"],
                color='red', label=f"Optimal Threshold = {optimal_threshold:.4f}", zorder=5)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.2)

    # Plot Accuracy vs Threshold
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, accuracies, label="Accuracy vs Threshold", linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--',
                label=f"Optimal Threshold = {optimal_threshold:.4f} (Accuracy = {max_accuracy:.4f})")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Threshold")
    plt.legend()
    plt.grid(alpha=0.2)

    # Show the plots
    plt.tight_layout()
    plt.show()


# Run the optimization method
file_path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_Mage_d3.csv"
true_label_column = 'AI_written'
pred_columns = [
    'roberta_large_openai_detector_probability',
]

results = calculate_pr_based_accuracy_threshold(file_path, true_label_column, pred_columns)
print(results)

# Plot the metrics
plot_pr_curve_with_accuracy(results)
