import pandas as pd

# Read datasets
dataset_detectors = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t3_features.csv"

test_dataset = pd.read_csv(dataset_detectors)

# test_dataset.shape
# List of AI probability classifiers
classifiers = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability']

def calculate_metrics_by_source(thresholds, data):
    results = {}
    for source in data['source'].unique():
        source_data = data[data['source'] == source].copy()
        
        # Create contingency table for each threshold combination
        for classifier in classifiers:
            if classifier not in source_data.columns:
                continue
            
            if source not in results:
                results[source] = {}
            if classifier not in results[source]:
                results[source][classifier] = {}
            
            for i, threshold in enumerate(thresholds, 1):
                source_data.loc[:, f'pred_{i}'] = source_data[classifier].apply(lambda x: 1 if x >= threshold else 0)
                source_data.loc[:, f'actual_{i}'] = source_data['AI_written'].apply(lambda x: 1 if x >= threshold else 0)
                
                TP = len(source_data[(source_data[f'pred_{i}'] == 1) & (source_data[f'actual_{i}'] == 1)])
                TN = len(source_data[(source_data[f'pred_{i}'] == 0) & (source_data[f'actual_{i}'] == 0)])
                FP = len(source_data[(source_data[f'pred_{i}'] == 1) & (source_data[f'actual_{i}'] == 0)])
                FN = len(source_data[(source_data[f'pred_{i}'] == 0) & (source_data[f'actual_{i}'] == 1)])
                
                # Calculate ACC and FPR
                ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
                FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                
                results[source][classifier][f'threshold_{i}'] = {'ACC': ACC, 'FPR': FPR}
    
    return results

# Example usage
thresholds = [0.5, 0.998441517353058] # we can try 0.9995323419570924 from FPR < 0.01 as well.

# Calculate metrics by source and classifier for the specified thresholds
metrics_by_source = calculate_metrics_by_source(thresholds, test_dataset)

print("\nMetrics by Source and Classifier:")
for source, classifiers_metrics in metrics_by_source.items():
    for classifier, metrics in classifiers_metrics.items():
        for threshold, metric in metrics.items():
            print(f"{source} & {classifier} & {threshold} & ACC: {metric['ACC']:.2f} & FPR: {metric['FPR']:.2f} \\\\")
