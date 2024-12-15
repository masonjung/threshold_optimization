import pandas as pd
import numpy as np

# Load dataset
dataset_detectors = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t4_features.csv"
df = pd.read_csv(dataset_detectors)


# Classifiers and thresholds
classifiers = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability', "GPT4o-mini_probability"]
thresholds = [0.5, 0.998441517353058] # add this for Precision-Recall curve 0.0002591597149148

# Create feature-based groups
length_groups = pd.cut(
    df['text_length'].fillna(-1),
    bins=[-1, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str)

formality_groups = df['formality'].apply(lambda x: 'formal' if x > 50 else 'informal').astype(str)
sentiment_groups = df['sentiment_label'].fillna('neutral').astype(str)
personality_groups = df['personality'].fillna('unknown').astype(str)

df['length_group'] = length_groups
df['formality_group'] = formality_groups
df['sentiment_group'] = sentiment_groups
df['personality_group'] = personality_groups


def calculate_metrics_by_group(data, group_col, thresholds, classifiers):
    """
    Calculates metrics by a specific group column (e.g., formality_group).
    Returns: results[group_value][classifier][threshold_key] = {TP, TN, FP, FN, etc.}
    """
    results = {}
    unique_groups = data[group_col].unique()
    for grp in unique_groups:
        grp_data = data[data[group_col] == grp].copy()
        
        if grp_data.empty:
            continue
        
        for classifier in classifiers:
            if classifier not in grp_data.columns:
                continue
            
            if grp not in results:
                results[grp] = {}
            if classifier not in results[grp]:
                results[grp][classifier] = {}
            
            for i, threshold in enumerate(thresholds, 1):
                pred_col = f'pred_{i}'
                actual_col = f'actual_{i}'
                
                grp_data[pred_col] = grp_data[classifier].apply(lambda x: 1 if x >= threshold else 0)
                grp_data[actual_col] = grp_data['AI_written'].apply(lambda x: 1 if x == 1 else 0)
                
                TP = len(grp_data[(grp_data[pred_col] == 1) & (grp_data[actual_col] == 1)])
                TN = len(grp_data[(grp_data[pred_col] == 0) & (grp_data[actual_col] == 0)])
                FP = len(grp_data[(grp_data[pred_col] == 1) & (grp_data[actual_col] == 0)])
                FN = len(grp_data[(grp_data[pred_col] == 0) & (grp_data[actual_col] == 1)])
                
                denom = TP + TN + FP + FN
                ACC = (TP + TN) / denom if denom > 0 else 0
                FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
                PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
                NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
                
                results[grp][classifier][f'threshold_{i}'] = {
                    'TP': TP,
                    'TN': TN,
                    'FP': FP,
                    'FN': FN,
                    'ACC': ACC,
                    'FPR': FPR,
                    'FNR': FNR,
                    'PPV': PPV,
                    'NPV': NPV
                }
    return results

def calculate_fairness_discrepancy(metrics_dict, classifiers, thresholds):
    """
    Calculates fairness metrics (TPR, FPR, TNR, FNR) for the given classifiers and thresholds.
    Returns the metric with the greatest disparity for each classifier and threshold.
    """
    fairness_metrics = ['TPR', 'FPR', 'TNR', 'FNR']
    discrepancies = {}
    groups = list(metrics_dict.keys())
    if len(groups) < 2:
        return discrepancies  # Not enough subgroups to measure discrepancy
    
    for classifier in classifiers:
        for i, threshold in enumerate(thresholds, 1):
            threshold_key = f'threshold_{i}'
            metric_discrepancies = {metric: [] for metric in fairness_metrics}
            
            for grp in groups:
                clf_data = metrics_dict.get(grp, {}).get(classifier, {})
                if threshold_key in clf_data:
                    TP = clf_data[threshold_key]['TP']
                    FP = clf_data[threshold_key]['FP']
                    FN = clf_data[threshold_key]['FN']
                    TN = clf_data[threshold_key]['TN']
                    
                    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
                    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
                    
                    metric_discrepancies['TPR'].append(TPR)
                    metric_discrepancies['FPR'].append(FPR)
                    metric_discrepancies['TNR'].append(TNR)
                    metric_discrepancies['FNR'].append(FNR)
            
            # Calculate discrepancies for each metric
            metric_disparities = {metric: (max(values) - min(values)) if values else 0 
                                  for metric, values in metric_discrepancies.items()}
            
            # Find the greatest discrepancy
            max_discrepancy_metric = max(metric_disparities, key=metric_disparities.get)
            max_discrepancy_value = metric_disparities[max_discrepancy_metric]
            
            if classifier not in discrepancies:
                discrepancies[classifier] = {}
            discrepancies[classifier][threshold_key] = {
                'Greatest Discrepancy': max_discrepancy_value,
                'Metric': max_discrepancy_metric
            }
    return discrepancies

# Calculate fairness metrics and statistical discrepancies
feature_cols = {
    'Formality': 'formality_group',
    'Length': 'length_group',
    'Sentiment': 'sentiment_group',
    'Personality': 'personality_group'
}

output_lines = []
for source in df['source'].unique():
    source_data = df[df['source'] == source]
    output_lines.append(f"\n\n=== Dataset (Source): {source} ===")
    
    for feature_name, group_col in feature_cols.items():
        output_lines.append(f"\n=== Feature: {feature_name} ===")
        
        # Calculate metrics for the feature group
        feature_metrics = calculate_metrics_by_group(source_data, group_col, thresholds, classifiers)
        
        # Calculate fairness discrepancies
        discrepancies = calculate_fairness_discrepancy(feature_metrics, classifiers, thresholds)
        
        # Log discrepancies for each classifier and threshold
        for classifier, discrepancy_data in discrepancies.items():
            output_lines.append(f"Classifier: {classifier}")
            for threshold_key, values in discrepancy_data.items():
                output_lines.append(f"  {threshold_key}: Greatest Discrepancy = {values['Greatest Discrepancy']:.3f} (Metric: {values['Metric']})")

# Save results to a text file
output_file = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\fairness_discrepancies.txt"
with open(output_file, "w") as file:
    file.write("\n".join(output_lines))

print(f"Results saved to {output_file}")
