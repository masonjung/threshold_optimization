import pandas as pd
import numpy as np

# Load dataset
dataset_detectors = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t3_features.csv"
df = pd.read_csv(dataset_detectors)

classifiers = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability']
thresholds = [0.5, 0.998441517353058, 0.0002591597149148]

# Create feature-based groups
length_groups = pd.cut(
    df['text_length'].fillna(-1),
    bins=[-1, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str)

formality_groups = pd.cut(
    df['formality'].fillna(-1),
    bins=[-1, 50, np.inf],
    labels=['informal', 'formal']
).astype(str)

sentiment_groups = df['sentiment_label'].fillna('neutral').astype(str)
personality_groups = df['personality'].fillna('unknown').astype(str)

df['length_group'] = length_groups
df['formality_group'] = formality_groups
df['sentiment_group'] = sentiment_groups
df['personality_group'] = personality_groups

def calculate_metrics_by_group(data, group_col, thresholds, classifiers):
    """
    Calculates metrics by a specific group column (e.g., formality_group).
    Returns: results[group_value][classifier][threshold_key] = {ACC, FPR, TP, TN, FP, FN}
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
                
                results[grp][classifier][f'threshold_{i}'] = {
                    'ACC': ACC,
                    'FPR': FPR,
                    'TP': TP,
                    'TN': TN,
                    'FP': FP,
                    'FN': FN
                }
    return results

def print_subgroup_discrepancies(metrics_dict, feature_name, classifiers, thresholds):
    """
    Calculates and prints discrepancies in ACC and FPR between subgroups.
    For binary subgroups (e.g., informal/formal), it's just the difference.
    For multiple subgroups, we use max-min difference.
    """
    print(f"\n=== {feature_name} Subgroup Discrepancies ===")
    # Extract all groups
    groups = list(metrics_dict.keys())
    if len(groups) < 2:
        print("Not enough subgroups to measure discrepancy.")
        return
    
    for classifier in classifiers:
        for i, threshold in enumerate(thresholds, 1):
            threshold_key = f"threshold_{i}"
            acc_values = []
            fpr_values = []
            
            for grp in groups:
                clf_data = metrics_dict.get(grp, {}).get(classifier, {})
                if threshold_key in clf_data:
                    acc_values.append(clf_data[threshold_key]['ACC'])
                    fpr_values.append(clf_data[threshold_key]['FPR'])
            
            if len(acc_values) < 2:
                # Not enough data points across groups
                continue
            
            # Discrepancy as max-min
            acc_discrepancy = max(acc_values) - min(acc_values)
            fpr_discrepancy = max(fpr_values) - min(fpr_values)
            
            print(f"Classifier: {classifier}, Threshold: {thresholds[i-1]:.4f}")
            print(f"ACC Discrepancy (Max-Min): {acc_discrepancy:.3f}")
            print(f"FPR Discrepancy (Max-Min): {fpr_discrepancy:.3f}")

# Now we loop over each dataset (source) and calculate discrepancies for each feature:
feature_cols = {
    'Formality': 'formality_group',
    'Length': 'length_group',
    'Sentiment': 'sentiment_group',
    'Personality': 'personality_group'
}

for source in df['source'].unique():
    source_data = df[df['source'] == source]
    print(f"\n\n=== Dataset (Source): {source} ===")
    for feature_name, group_col in feature_cols.items():
        feature_metrics = calculate_metrics_by_group(source_data, group_col, thresholds, classifiers)
        print_subgroup_discrepancies(feature_metrics, feature_name, classifiers, thresholds)
