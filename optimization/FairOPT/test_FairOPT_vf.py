import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
import os

# Minseok's path
#path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
#train_path = path+"\\train_features.csv"

# Cyntia's path
path = 'C://Users//Cynthia//Documents//MIT//datasets'
train_path = path+'//train_features.csv'
test_path = path+'//test_t4_features.csv'
results_thresholds_path =  path+'//results_thresholds.json'
results_path =  path+'//results.json'

acceptable_disparities = [1.00, 0.9, 0.8, 0.7,0.6, 0.5, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]


test_df = pd.read_csv(test_path)


with open(results_thresholds_path, 'r') as f:
    disparities_results = json.load(f)

#################################################
# Get the metrics with the optimized thresholds
#################################################

#thresholds = {k: v['thresholds_opt'] for k, v in disparities_results.items()}
thresholds_disparity = {disparity: results["thresholds_opt"] for disparity, results in disparities_results.items()}
#print(thresholds)

unique_sources = test_df['source'].unique()
detector_probabilities = [
    'radar_probability',
    'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    'GPT4o-mini_probability'
]
features_columns = ['length_personality']
test_df['length_personality'] = test_df['length_label'] + '_' + test_df['personality']

if os.path.exists(results_path):
    os.remove(results_path)

with open(results_path, 'a') as results_file:
    for col in features_columns:
        for acceptable_disparity in acceptable_disparities:
            test_df['y_true'] = test_df['AI_written']  # True labels
            thresholds = thresholds_disparity[str(acceptable_disparity)]

            # Store results for each source across all detectors
            source_results = {source: {'accuracy': [], 'f1': [], 'fpr': [], 'ber': []} for source in unique_sources}
            overall_results = {'accuracy': [], 'f1': [], 'fpr': [], 'ber': []}

            for source in unique_sources:
                for detector in detector_probabilities:
                    
                    test_df_filtered = test_df[test_df[col].isin(thresholds.keys())]
                    test_df_filtered['y_pred_proba'] = test_df_filtered[detector] # Predicted labels
                    test_df_filtered['y_pred'] = test_df_filtered.apply(
                        lambda row: 1 if row['y_pred_proba'] > thresholds[row[col]] else 0, axis=1
                    )
                    source_df = test_df_filtered[test_df_filtered['source'] == source]

                    accuracy = accuracy_score(source_df['y_true'], source_df['y_pred'])
                    f1 = f1_score(source_df['y_true'], source_df['y_pred'], zero_division=1)
                    fpr = np.sum((source_df['y_pred'] == 1) & (source_df['y_true'] == 0)) / np.sum(source_df['y_true'] == 0)
                    fnr = np.sum((source_df['y_pred'] == 0) & (source_df['y_true'] == 1)) / np.sum(source_df['y_true'] == 1)
                    ber = 0.5 * (fpr + fnr)

                    # Store individual results
                    source_results[source]['accuracy'].append(accuracy)
                    source_results[source]['f1'].append(f1)
                    source_results[source]['fpr'].append(fpr)
                    source_results[source]['ber'].append(ber)

                    overall_results['accuracy'].append(accuracy)
                    overall_results['f1'].append(f1)
                    overall_results['fpr'].append(fpr)
                    overall_results['ber'].append(ber)

                    results_file.write(f"\nPerformance for Source: {source}, Detector: {detector}, Disparity: {acceptable_disparity}\n")
                    results_file.write(f"Accuracy: {accuracy:.4f}\n")
                    results_file.write(f"F1: {f1:.4f}\n")
                    results_file.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
                    results_file.write(f"Balanced Error (BER): {ber:.4f}\n")

            # Aggregate results per source
            for source in unique_sources:
                avg_accuracy = np.mean(source_results[source]['accuracy'])
                avg_f1 = np.mean(source_results[source]['f1'])
                avg_fpr = np.mean(source_results[source]['fpr'])
                avg_ber = np.mean(source_results[source]['ber'])

                results_file.write(f"\nOverall Performance for Source: {source}, Disparity: {acceptable_disparity}\n")
                results_file.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
                results_file.write(f"Average F1: {avg_f1:.4f}\n")
                results_file.write(f"Average False Positive Rate (FPR): {avg_fpr:.4f}\n")
                results_file.write(f"Average Balanced Error (BER): {avg_ber:.4f}\n")

            # Compute overall performance
            overall_avg_accuracy = np.mean(overall_results['accuracy'])
            overall_avg_f1 = np.mean(overall_results['f1'])
            overall_avg_fpr = np.mean(overall_results['fpr'])
            overall_avg_ber = np.mean(overall_results['ber'])

            results_file.write(f"\nFinal Overall Performance for Entire Dataset, Disparity: {acceptable_disparity}\n")
            results_file.write(f"Average Accuracy: {overall_avg_accuracy:.4f}\n")
            results_file.write(f"Average F1: {overall_avg_f1:.4f}\n")
            results_file.write(f"Average False Positive Rate (FPR): {overall_avg_fpr:.4f}\n")
            results_file.write(f"Average Balanced Error (BER): {overall_avg_ber:.4f}\n")
