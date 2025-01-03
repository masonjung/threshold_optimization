import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import FairOPT

path = 'C://Users//Cynthia//Documents//IpParis_C//MIT//datasets'
test_path = path+'//test_t4_features.csv'

test_dataset = pd.read_csv(test_path)

# need to apply the generated thresold to the test dataset
# Load test dataset

# Split test dataset by 'source'
unique_sources = test_dataset['source'].unique()
detector_probabilities = ['radar_probability', 'roberta_base_openai_detector_probability', 'roberta_large_openai_detector_probability', 'GPT4o-mini_probability']


def test_thresholds(test_dataset, source, thresholds, acceptable_disparity, count_group):
#for source in unique_sources:
    source_dataset = test_dataset[test_dataset['source'] == source]

    # Split by different detector probabilities    
    for detector in detector_probabilities:
        if detector not in source_dataset.columns:
            continue

        # Prepare test dataset groups
        test_length_groups = pd.cut(
            source_dataset['text_length'],
            bins=[0, 1000, 2500, np.inf],
            labels=['short', 'medium', 'long']
        ).astype(str).values

        test_sentiment_groups = source_dataset['sentiment_label'].astype(str).values

        test_formality_groups = pd.cut(
            source_dataset['formality'],
            bins=[0, 50, np.inf],
            labels=['informal', 'formal']
        ).astype(str).values

        test_personality_groups = source_dataset['personality'].astype(str).values

        # Combine groups into a single group label for test dataset
        test_groups = pd.Series([
            f"{length}_{formality}_{sentiment}_{personality}"
            for length, formality, sentiment, personality in zip(test_length_groups, test_formality_groups, test_sentiment_groups, test_personality_groups)
        ]).values

        # Prepare true labels and predicted probabilities for test dataset
        test_y_true = source_dataset['AI_written']        
        test_y_pred_proba = source_dataset[detector].values

        # Apply optimized thresholds to test dataset
        test_y_pred = np.zeros_like(test_y_true)
        for group in np.unique(test_groups):
            group_indices = (test_groups == group)
            threshold = thresholds.get(group, 0.5)  # Default to 0.5 if group not found
            test_y_pred[group_indices] = test_y_pred_proba[group_indices] >= threshold

        # Calculate and print performance metrics for the current source and detector
        test_accuracy = accuracy_score(test_y_true, test_y_pred)
        test_fpr = np.sum((test_y_pred == 1) & (test_y_true == 0)) / np.sum(test_y_true == 0)
        test_fnr = np.sum((test_y_pred == 0) & (test_y_true == 1)) / np.sum(test_y_true == 1)
        balanced_error = (test_fpr + test_fnr) / 2 # BER = ((FN/TP+FN) + (FP/TN+FP)) / 2

        print(f"\nPerformance for Source: {source}, Detector: {detector}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"False Positive Rate (FPR): {test_fpr:.4f}")
        print(f"Balanced Error (BER): {balanced_error:.4f}")

        # store the printed things in txt and threshold for each
        results_path = path+f"//results_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        
        with open(results_path, "a") as f:
            f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"False Positive Rate (FPR): {test_fpr:.4f}\n")
            f.write(f"Balanced Error (BER): {balanced_error:.4f}\n")
            # f.write(f"Thresholds:\n")
            # for group, threshold in thresholds.items():
            #     f.write(f"Group: {group}, Threshold: {threshold:.7f}\n")            


acceptable_disparities =  [1, 0.2, 0.1] #[1, 0.2, 0.1, 0.01, 0.001]
num_groups = 10
count_groups = [str(i).zfill(2) for i in range(1, num_groups+1)]

for count_group in count_groups:
    for acceptable_disparity in acceptable_disparities:
        results_path = path+f"//results_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        if os.path.exists(results_path):
            os.remove(results_path)
            
        print("\n"+"="*100)
        print(f"Results for acceptable_disparity = {acceptable_disparity}:\n")
        # Load optimized thresholds
        #file_path = path+f"//thresholds_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        file_path = path+f"//thresholds_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        with open(file_path, 'r') as file:
            thresholds = {}
            for line in file:
                group, threshold = line.strip().split(", ")
                group = group.split(": ")[1]
                threshold = float(threshold.split(": ")[1])
                thresholds[group] = threshold

        #print(thresholds)
        # Apply thresholds to test dataset
        for source in unique_sources:
            test_thresholds(test_dataset, source, thresholds, acceptable_disparity, count_group)
            
        results_path = path+f"//results_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        print(f"Results for disparity: {acceptable_disparity:.4f} have been saved to:", results_path)