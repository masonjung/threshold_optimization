import pandas as pd
import numpy as np

# Load dataset
dataset_detectors = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t4_features.csv"
df = pd.read_csv(dataset_detectors)

# Classifiers and thresholds
classifiers = ['roberta_large_openai_detector_probability', 'radar_probability', 
               'roberta_base_openai_detector_probability', "GPT4o-mini_probability"]
thresholds = [0.5, 0.998441517353058]

# Create feature-based groups
df['length_group'] = pd.cut(
    df['text_length'].fillna(-1),
    bins=[-1, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str)

df['formality_group'] = df['formality'].apply(lambda x: 'formal' if x > 50 else 'informal').astype(str)
df['sentiment_group'] = df['sentiment_label'].fillna('neutral').astype(str)
df['personality_group'] = df['personality'].fillna('unknown').astype(str)

def calculate_metrics_by_group(data, group_col, thresholds, classifiers):
    """
    Calculates classification metrics by group (e.g., length_group).
    Returns: results[group_value][classifier][threshold_key] = {TP, TN, FP, FN, ACC, FPR, FNR}
    """
    results = {}
    unique_groups = data[group_col].unique()

    for grp in unique_groups:
        grp_data = data[data[group_col] == grp]
        if grp_data.empty:
            continue

        for classifier in classifiers:
            if classifier not in grp_data.columns:
                continue  # Skip classifiers not in the dataset

            if grp not in results:
                results[grp] = {}
            if classifier not in results[grp]:
                results[grp][classifier] = {}

            for i, threshold in enumerate(thresholds, 1):
                predictions = grp_data[classifier].apply(lambda x: 1 if x >= threshold else 0)
                actuals = grp_data['AI_written'].apply(lambda x: 1 if x == 1 else 0)

                TP = ((predictions == 1) & (actuals == 1)).sum()
                TN = ((predictions == 0) & (actuals == 0)).sum()
                FP = ((predictions == 1) & (actuals == 0)).sum()
                FN = ((predictions == 0) & (actuals == 1)).sum()

                denom = TP + TN + FP + FN
                results[grp][classifier][f'threshold_{i}'] = {
                    'TP': TP,
                    'TN': TN,
                    'FP': FP,
                    'FN': FN,
                    'ACC': (TP + TN) / denom if denom > 0 else 0,
                    'FPR': FP / (FP + TN) if (FP + TN) > 0 else 0,
                    'FNR': FN / (FN + TP) if (FN + TP) > 0 else 0
                }
    return results

def calculate_statistical_discrepancy(metrics_dict, classifiers, thresholds):
    """
    Calculates the statistical discrepancy across groups for each classifier and threshold.
    """
    discrepancies = {}
    groups = list(metrics_dict.keys())

    if len(groups) < 2:
        return discrepancies  # Discrepancy calculation requires at least two groups

    for classifier in classifiers:
        for i, threshold in enumerate(thresholds, 1):
            threshold_key = f'threshold_{i}'
            metric_rates = {'FNR': []}

            for grp in groups:
                group_metrics = metrics_dict.get(grp, {}).get(classifier, {}).get(threshold_key, {})
                for metric in metric_rates:
                    rate = group_metrics.get(metric, 0)
                    metric_rates[metric].append(rate)

            max_discrepancies = {metric: max(rates) - min(rates) if rates else 0 
                                 for metric, rates in metric_rates.items()}
            max_discrepancy = max(max_discrepancies.values())

            if classifier not in discrepancies:
                discrepancies[classifier] = {}
            discrepancies[classifier][threshold_key] = {
                **max_discrepancies,
                'Max Discrepancy': max_discrepancy
            }
    return discrepancies

# Group feature columns
feature_cols = {
    'Formality': 'formality_group',
    'Length': 'length_group',
    'Sentiment': 'sentiment_group',
    'Personality': 'personality_group'
}

# Calculate metrics and discrepancies for each source
output_lines = []
for source in df['source'].unique():
    source_data = df[df['source'] == source]
    output_lines.append(f"\n\n=== Dataset (Source): {source} ===")

    for feature_name, group_col in feature_cols.items():
        output_lines.append(f"\n=== Feature: {feature_name} ===")
        
        # Calculate metrics for the feature group
        feature_metrics = calculate_metrics_by_group(source_data, group_col, thresholds, classifiers)
        
        # Calculate discrepancies
        discrepancies = calculate_statistical_discrepancy(feature_metrics, classifiers, thresholds)
        
        # Log discrepancies
        for classifier, discrepancy_data in discrepancies.items():
            output_lines.append(f"Classifier: {classifier}")
            for threshold_key, values in discrepancy_data.items():
                output_lines.append(f"  {threshold_key}: Max Discrepancy = {values['Max Discrepancy']:.4f}")
                output_lines.append(f"  {threshold_key}: FNR Discrepancy = {values['FNR']:.4f}")
                for metric, value in values.items():
                    if metric != 'Max Discrepancy':
                        output_lines.append(f"    {metric}: {value:.4f}")

# Save results to a file
output_file = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\statistical_discrepancies_with_acc_fpr.txt"
with open(output_file, "w") as file:
    file.write("\n".join(output_lines))

print(f"Results saved to {output_file}")
