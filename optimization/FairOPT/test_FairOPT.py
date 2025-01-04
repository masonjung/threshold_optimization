import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import FairOPT


# Minseok's path
path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
test_path = path+"\\test_t4_features.csv"


# Ctynia's path
# path = 'C://Users//Cynthia//Documents//MIT//datasets'
# test_path = path+'//test_t4_features.csv'

test_dataset = pd.read_csv(test_path)

# need to apply the generated thresold to the test dataset
# Load test dataset

# Split test dataset by 'source'
unique_sources = test_dataset['source'].unique()
detector_probabilities = ['radar_probability', 'roberta_base_openai_detector_probability', 'roberta_large_openai_detector_probability', 'GPT4o-mini_probability']


# Length-based groups
length_groups = pd.cut(
    test_dataset['text_length'].dropna(),
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values

test_dataset['length_feature'] = pd.cut(
    test_dataset['text_length'].dropna(),
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values


# Formality-based groups
formality_groups = pd.cut(
    test_dataset['formality'].dropna(),
    bins=[0, 50, np.inf],
    labels=['informal', 'formal']
).astype(str).values
formality_groups = [x for x in formality_groups if x != 'nan']

test_dataset['formality_feature'] = pd.cut(
    test_dataset['formality'].dropna(),
    bins=[0, 50, np.inf],
    labels=['informal', 'formal']
).astype(str).values


# Sentiment and personality groups (ensure no NaN)
sentiment_groups = test_dataset['sentiment_label'].fillna('neutral').astype(str).values
personality_groups = test_dataset['personality'].fillna('unknown').astype(str).values

uni_length_groups = np.unique(length_groups).tolist()
uni_formality_groups = np.unique(formality_groups).tolist()
uni_sentiment_groups = np.unique(sentiment_groups).tolist()
uni_personality_groups = np.unique(personality_groups).tolist()



def generate_groups(group_words, df):
    # Check which group each word belongs to and identify the corresponding columns
    columns_to_include = []
    if any(word in uni_length_groups for word in group_words):
        columns_to_include.append('length_feature')
    if any(word in uni_formality_groups for word in group_words):
        columns_to_include.append('formality_feature')
    if any(word in uni_sentiment_groups for word in group_words):
        columns_to_include.append('sentiment_label')
    if any(word in uni_personality_groups for word in group_words):
        columns_to_include.append('personality')

    # Combine the values of the identified columns, separated by "_"
    return df[columns_to_include].apply(lambda row: '_'.join(row.astype(str)), axis=1)

def test_thresholds(test_dataset, source, thresholds, acceptable_disparity, count_group, test_groups_unique, groups_column):
#for source in unique_sources:
    source_dataset = test_dataset[test_dataset['source'] == source]
    source_groups_column = groups_column[test_dataset['source'] == source]

    # Split by different detector probabilities    
    for detector in detector_probabilities:
        if detector not in source_dataset.columns:
            continue

        # Prepare true labels and predicted probabilities for test dataset
        test_y_true = source_dataset['AI_written']        
        test_y_pred_proba = source_dataset[detector].values

        # Apply optimized thresholds to test dataset
        test_y_pred = np.zeros_like(test_y_true)
        for group in test_groups_unique:
            group_indices = (source_groups_column == group)
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


acceptable_disparities =  [1, 0.5, 0.2, 0.1, 0.01, 0.001] # [1, 0.5, 0.2, 0.1, 0.01, 0.001]
num_groups = 10
count_groups = [str(i).zfill(2) for i in range(8, num_groups+1)]


for count_group in count_groups:
    for acceptable_disparity in acceptable_disparities:
        results_path = path+f"\\results_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        if os.path.exists(results_path):
            os.remove(results_path)
            
        print("\n"+"="*100)
        print(f"Results for acceptable_disparity = {acceptable_disparity}:\n")
        # Load optimized thresholds
        #file_path = path+f"//thresholds_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        file_path = path+f"\\thresholds_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        with open(file_path, 'r') as file:
            thresholds = {}
            test_groups_unique = []
            for line in file:
                group, threshold = line.strip().split(", ")
                group = group.split(": ")[1]
                threshold = float(threshold.split(": ")[1])
                thresholds[group] = threshold
                test_groups_unique.append(group)
        
        groups_column = generate_groups(test_groups_unique[0].split("_"), test_dataset)

        # Apply thresholds to test dataset
        for source in unique_sources:
            test_thresholds(test_dataset, source, thresholds, acceptable_disparity, count_group, test_groups_unique, groups_column)
            
        #results_path = path+f"//results_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        results_path = path+f"\\results_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
        print(f"Results for group: {str(count_group).zfill(2)} and disparity: {acceptable_disparity:.4f} have been saved to:", results_path)