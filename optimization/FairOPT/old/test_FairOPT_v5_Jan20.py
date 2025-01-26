import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import FairOPT


# Minseok's path
#path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
#test_path = path+"\\test_t4_features.csv"


# Cynthia's path
path = 'C://Users//Cynthia//Documents//IpParis_C//MIT//datasets'
test_path = path+'//test_t4_features.csv'

test_df = pd.read_csv(test_path)

# Split test dataset by 'source'
unique_sources = test_df['source'].unique()
detector_probabilities = ['radar_probability', 'roberta_base_openai_detector_probability', 'roberta_large_openai_detector_probability', 'GPT4o-mini_probability']
#features_columns  = ['length_label', 'personality', 'length_personality']
features_columns  = ['length_personality']
acceptable_disparities =  [ 1.00, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]


test_df['length_personality'] = test_df['length_label'] + '_' + test_df['personality']

for col in features_columns:
    for acceptable_disparity in acceptable_disparities:
        
        files = os.listdir(path)
        #for file in files:
        #    print(file)
        
        #print(files)
        matching_file = [f for f in files if f.startswith('threshold') and col in f and str(acceptable_disparity).replace('.', '_').ljust(4, '0') in f]
        #matching_file = [f for f in files if col in f and str(acceptable_disparity).replace('.', '_').ljust(4, '0') in f]
        #for matching_file in matching_files:
        #    print(f"Matching file: {matching_file}")
        if not matching_file:
            break
              
        with open(os.path.join(path, matching_file[0]), 'r') as file:
            content = file.read()
            print(f"Content of {matching_file}:\n{content}\n")

            thresholds = {}
            lines = content.split('\n')
            for line in lines:
                if line.strip():
                    group, threshold = line.split(', Threshold: ')
                    group = group.split(': ')[1].strip()
                    threshold = float(threshold.strip())
                    thresholds[group] = threshold
            
            print(f"Thresholds: {thresholds}")
            
        test_df['y_true'] = test_df['AI_written']  # True labels
        
        
        results_path = path + '//' + matching_file[0].replace('thresholds_', 'results_')
        print(results_path)
        if os.path.exists(results_path):
            os.remove(results_path)
                    
        with open(results_path, 'a') as results_file:               
            print("="*100)
            for source in unique_sources:
                for detector in detector_probabilities:
                    #print(f"Source: {source}, Detector: {detector}")
                    test_df = test_df[test_df[col].isin(thresholds.keys())]
                    test_df['y_pred_proba'] = test_df[detector] # Predicted labels
                    test_df['y_pred'] = test_df.apply(
                        lambda row: 1 if row['y_pred_proba'] > thresholds[row[col]] else 0, axis=1
                    )
                    source_df = test_df[(test_df['source'] == source)]
                    accuracy = accuracy_score(source_df['y_true'], source_df['y_pred'])
                    #print(f"Accuracy for source {source} and detector {detector}: {accuracy}")
                    fpr = np.sum((source_df['y_pred'] == 1) & (source_df['y_true'] == 0)) / np.sum(source_df['y_true'] == 0)
                    fnr = np.sum((source_df['y_pred'] == 0) & (source_df['y_true'] == 1)) / np.sum(source_df['y_true'] == 1)
                    ber = 0.5 * (fpr + fnr)
                    #print(f"False Positive Rate for source {source} and detector {detector}: {fpr}")
                    #print(f"Balanced Error Rate for source {source} and detector {detector}: {ber}")
                    
                    print(f"\nPerformance for Source: {source}, Detector: {detector}")
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"False Positive Rate (FPR): {fpr:.4f}")
                    print(f"Balanced Error (BER): {ber:.4f}")
            
                    results_file.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
                    results_file.write(f"Accuracy: {accuracy:.4f}\n")
                    results_file.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
                    results_file.write(f"Balanced Error (BER): {ber:.4f}\n")
            
        print("="*100)