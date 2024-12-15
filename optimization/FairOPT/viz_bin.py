import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df', 'y_true', 'y_pred_proba', 'groups', and 'thresholds' are already defined

# Step 1: Define function to compute group metrics for a single threshold
def compute_group_metrics(y_true, y_pred_proba, groups, threshold):
    group_metrics = {}
    unique_groups = np.unique(groups)

    # Apply the same threshold to all predictions
    y_pred = y_pred_proba >= threshold

    for group in unique_groups:
        indices = groups == group
        group_y_true = y_true[indices]
        group_y_pred = y_pred[indices]

        # Compute metrics
        acc = accuracy_score(group_y_true, group_y_pred)
        f1 = f1_score(group_y_true, group_y_pred, zero_division=1)
        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
        tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
        fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
        fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        group_metrics[group] = {
            'Accuracy': acc,
            'F1 Score': f1,
            'TPR': tpr,
            'FPR': fpr
        }

    return group_metrics

# Step 2: Compute metrics for static threshold (0.5)
static_threshold = 0.5  # Single threshold for all groups
static_metrics = compute_group_metrics(y_true, y_pred_proba, groups, static_threshold)

# Step 4: Compute metrics for optimized thresholds (group-specific)
# Assuming 'thresholds' is a dictionary with group-specific thresholds from your optimizer
def compute_group_metrics_with_group_thresholds(y_true, y_pred_proba, groups, thresholds):
    group_metrics = {}
    unique_groups = np.unique(groups)

    for group in unique_groups:
        indices = groups == group
        group_threshold = thresholds.get(group, 0.5)  # Default to 0.5 if group threshold is not available
        group_y_true = y_true[indices]
        group_y_pred_proba = y_pred_proba[indices]
        group_y_pred = group_y_pred_proba >= group_threshold

        # Compute metrics
        acc = accuracy_score(group_y_true, group_y_pred)
        f1 = f1_score(group_y_true, group_y_pred, zero_division=1)
        tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
        tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
        fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
        fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        group_metrics[group] = {
            'Accuracy': acc,
            'F1 Score': f1,
            'TPR': tpr,
            'FPR': fpr
        }

    return group_metrics

# Compute metrics for optimized thresholds
optimized_metrics = compute_group_metrics_with_group_thresholds(y_true, y_pred_proba, groups, thresholds)

# Step 5: Compile metrics into a DataFrame
def compile_metrics_to_df(static_metrics, optimized_metrics):
    data = []
    for group in static_metrics.keys():
        data.append({
            'Group': group,
            'Method': 'Static (0.5)',
            **static_metrics[group]
        })
        data.append({
            'Group': group,
            'Method': 'Optimized',
            **optimized_metrics[group]
        })
    df_metrics = pd.DataFrame(data)
    return df_metrics

df_metrics = compile_metrics_to_df(static_metrics, optimized_metrics)

# Compute overall metrics for comparison
overall_static_metrics = compute_group_metrics(y_true, y_pred_proba, groups, static_threshold)
overall_optimized_metrics = compute_group_metrics_with_group_thresholds(y_true, y_pred_proba, groups, thresholds)

overall_data = []
overall_data.append({
    'Group': 'Overall',
    'Method': 'Static (0.5)',
    **overall_static_metrics[list(overall_static_metrics.keys())[0]]
})
overall_data.append({
    'Group': 'Overall',
    'Method': 'Optimized',
    **overall_optimized_metrics[list(overall_optimized_metrics.keys())[0]]
})
df_overall_metrics = pd.DataFrame(overall_data)
df_metrics = pd.concat([df_metrics, df_overall_metrics], ignore_index=True)

# Display the DataFrame for verification
print(df_metrics.head())

# Step 6: Create bar charts
sns.set(style="whitegrid")
performance_metrics = ['Accuracy', 'F1 Score', 'TPR', 'FPR']

for metric in performance_metrics:
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='Group',
        y=metric,
        hue='Method',
        data=df_metrics,
        palette='Set1'
    )
    plt.title(f'Comparison of {metric} Across Groups and Methods')
    plt.xlabel('Group')
    plt.ylabel(metric)
    plt.xticks(rotation=90)

    # Create legend to show overall performance instead of annotating bars
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for method in df_metrics['Method'].unique():
        overall_metric_value = df_overall_metrics[df_overall_metrics['Method'] == method][metric].values[0]
        new_labels.append(f'{method} (Overall {metric}: {overall_metric_value:.4f})')
    ax.legend(handles, new_labels, title='Method')

    plt.tight_layout()
    plt.show()
