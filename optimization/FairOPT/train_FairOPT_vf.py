import pandas as pd
import numpy as np
import FairOPT_vf as FairOPT
from sklearn.metrics import accuracy_score, f1_score
import time
import json
import os
start_time = time.time()

# Minseok's path
#path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
#train_path = path+"\\train_features.csv"

# Cyntia's path
path = 'C://Users//Cynthia//Documents//MIT//datasets'
train_path = path+'//train_features.csv'
test_path = path+'//test_t4_features.csv'
results_thresholds_path =  path+'//results_thresholds.json'
results_path =  path+'//results.json'



##########################################
# Hyperparameters
##########################################

acceptable_disparities = [1.00, 0.9, 0.8, 0.7,0.6, 0.5, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]

learning_rate = 1e-3        # Adjusted learning rate. For threshold adjustment.
penalty=10                  # Penalty applied if the performance metrics score (acc and f1) is below the minimum threshold.
max_iteration_minimize = 15 # Maximum number of iterations of the "minimize" function, which objective is to minimize the loss function.

num_features = [1, 2]  # [1, 2, 3, 4]
frac = 1.0             # sample fraction of dataset

train_df = pd.read_csv(train_path)
train_df['length_personality'] = train_df['length_label'] + '_' + train_df['personality']
#features_columns  = ['length_label', 'personality', 'length_personality']
features_columns  = ['length_personality']

# df contains a random sample of the rows from "train_df", stratified by the column ['AI_written'].
df = train_df.groupby(['AI_written'], group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(test_path)



##########################################
# Optimize thresholds
##########################################

# True labels and Predicted probabilities learned from one model
y_true = df['AI_written']  # True labels
y_pred_proba = df['roberta_large_openai_detector_probability'] # Predicted labels


for _, col in enumerate(features_columns):
    unique_values_group = np.unique(df[col])
    group_indices = {}
    
    for value in unique_values_group:
        group_indices[value] = (df[col]==value)
    
    optimizer = FairOPT.ThresholdOptimizer(
        y_true = y_true,
        y_pred_proba = y_pred_proba,
        
        group_indices = group_indices,
        acceptable_disparities = acceptable_disparities,
                
        learning_rate = learning_rate,
        penalty = penalty,
        max_iteration_minimize = max_iteration_minimize
    )

    # Optimize thresholds using gradient-based method
    disparities_results = optimizer.optimize()
    
#print(f'disparities_results: {disparities_results}')
end_time = time.time()
execution_time = end_time - start_time
minutes = execution_time // 60
seconds = execution_time % 60
print(f"The code took {minutes} minutes and {seconds:.2f} seconds to run.")


if os.path.exists(results_thresholds_path):
    os.remove(results_thresholds_path)
    
with open(results_thresholds_path, 'w') as f:
    json.dump(disparities_results, f, indent=4)
