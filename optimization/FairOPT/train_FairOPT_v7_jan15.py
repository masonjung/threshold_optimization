import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from itertools import combinations
from itertools import product
import FairOPT
import acc_f1_baseline

# Minseok's path
#path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
#train_path = path+"\\train_features.csv"


# Cyntia's path
path = 'C://Users//Cynthia//Documents//MIT//datasets'
train_path = path+'//train_features.csv'



# hyperparameters
acceptable_disparities =  [ 1.00, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]
max_iterations = 2.5*10**4
learning_rate = 10**-2
tolerance = 1e-2 #10**-5
#min_acc_threshold = 0 #0.402753 #0.4 #0.5
#min_f1_threshold = 0 # 0.275660 #0.4 #0.5
initial_thresholds_values = 0.5# 0.1
penalty=1 #10 #20  # Increase penalty to enforce stricter updates
num_features = [1, 2] #[1, 2, 3, 4]
frac = 1.0 # sample fraction of dataset
#perc_excluded = 1
min_data = 100 # 500
min_accuracy = 0.2

train_df = pd.read_csv(train_path)
train_df['length_personality'] = train_df['length_label'] + '_' + train_df['personality']
features_columns  = ['length_label', 'personality', 'length_personality']
df = train_df.groupby(['AI_written'], group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
df.reset_index(drop=True, inplace=True)



##########################################
# Obtaining the initial thresholds
##########################################

baseline_per_group = []
for i_col, col in enumerate(features_columns):
    labels = df[col].unique()
    for i_label, label in enumerate(labels):
        df_group = df[df[col] == label]
        y_true = df_group['AI_written']  # True labels
        y_pred_proba = df_group['roberta_large_openai_detector_probability'] # Predicted labels
        results_df, static_threshold, roc_threshold = acc_f1_baseline.evaluate_thresholds(y_true, y_pred_proba)
        
        static_row = results_df[results_df['threshold'] == static_threshold]
        roc_row = results_df[results_df['threshold'] == roc_threshold]
        baseline_per_group.append({
            'group': 'G_'+str(i_col+1),
            'column': col,
            'label': label,
            'static_threshold': static_threshold,
            'static_accuracy': static_row['accuracy'].iloc[0] if not static_row.empty else None,
            'static_f1': static_row['f1'].iloc[0] if not static_row.empty else None,
            'static_data_above_threshold': static_row['perc_below_threshold'].iloc[0] if not static_row.empty else None,
            'static_data_below_threshold': static_row['perc_above_threshold'].iloc[0] if not static_row.empty else None,
            'roc_threshold': roc_threshold,
            'roc_accuracy': roc_row['accuracy'].iloc[0] if not roc_row.empty else None,
            'roc_f1': roc_row['f1'].iloc[0] if not roc_row.empty else None,
            'roc_data_above_threshold': roc_row['perc_below_threshold'].iloc[0] if not roc_row.empty else None,
            'roc_data_below_threshold': roc_row['perc_above_threshold'].iloc[0] if not roc_row.empty else None,
        })

baseline_df = pd.DataFrame(baseline_per_group)
baseline_df = baseline_df.sort_values(by=['group', 'label'], ascending=True)
baseline_df = baseline_df.drop_duplicates(subset=['group', 'label'], keep='first')
baseline_df = baseline_df.reset_index(drop=True)

value_counts = train_df['length_personality'].value_counts().reset_index()
value_counts.columns = ['length_personality', 'count']
value_counts['percentage'] = (value_counts['count'] / len(train_df)) * 100

initial_thresholds = baseline_df.groupby(['group', 'column', 'label']).agg(
        min_accuracy=('static_accuracy', lambda x: min(x.min(), baseline_df.loc[x.index, 'roc_accuracy'].min())),
        min_f1=('static_f1', lambda x: min(x.min(), baseline_df.loc[x.index, 'roc_f1'].min()))
    ).reset_index()


#exclude_values = value_counts.loc[value_counts['percentage'] < perc_excluded, 'length_personality']
#exclude_values = value_counts.loc[value_counts['count'] < min_data, 'length_personality']
#exclude_values = initial_thresholds[initial_thresholds['min_accuracy'] < 0.2]['label']
exclude_values_1 = initial_thresholds.loc[initial_thresholds['min_accuracy'] < min_accuracy, 'label']
exclude_values_2 = value_counts.loc[value_counts['count'] < min_data, 'length_personality']
exclude_values = exclude_values_1.to_numpy().tolist() + exclude_values_2.to_numpy().tolist()
df = df[~df['length_personality'].isin(exclude_values)].reset_index(drop=True)
initial_thresholds = initial_thresholds[~initial_thresholds['label'].isin(exclude_values)].reset_index(drop=True)

print(df.shape)
print(initial_thresholds)
print(exclude_values)
#asd


##########################################
# Optimize thresholds
##########################################

def optimize_thresholds(y_true, y_pred_proba, groups, initial_thresholds, group_indices, acceptable_disparity, max_iterations, learning_rate, tolerance, penalty, min_acc_threshold, min_f1_threshold, group_column):#,path):
    
    optimizer = FairOPT.ThresholdOptimizer(
        y_true = y_true,
        y_pred_proba = y_pred_proba,
        groups = groups,
        initial_thresholds = initial_thresholds,
        group_indices = group_indices,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        acceptable_disparity=acceptable_disparity,  # Adjust based on your fairness criteria
        min_acc_threshold = min_acc_threshold,  # Set realistic minimum accuracy
        min_f1_threshold = min_f1_threshold,   # Set realistic minimum F1 score
        tolerance = tolerance,    # Decrease tolerance for stricter convergence criteria
        penalty = penalty, #20  # Increase penalty to enforce stricter updates
    )

    # Optimize thresholds using gradient-based method
    #thresholds, history, iteration = optimizer.optimize()
    thresholds, iteration, opt_learning_rate, opt_acc_dict, opt_f1_dict, is_convergence, delta = optimizer.optimize()
    
    if is_convergence:
        # Save thresholds to a file
        file_path = path+f"//thresholds_{group_column}_disparity_{str(acceptable_disparity).replace('.', '_').ljust(4, '0')}.txt"
        with open(file_path, 'w') as file:
            for group, threshold in thresholds.items():
                file.write(f"Group: {group}, Threshold: {threshold:.7f}\n")
        print("The optimized thresholds have been saved to:", file_path)

    # Move the results to the list
    optimized_thresholds_list = []
    for group, threshold in thresholds.items():
        optimized_thresholds_list.append({'group': group, 'threshold': threshold})

    return optimized_thresholds_list, iteration, opt_learning_rate, opt_acc_dict, opt_f1_dict, is_convergence, delta



##########################################
# Write the results in .txt files
##########################################


# True labels and Predicted probabilities learned from one model
y_true = df['AI_written']  # True labels
y_pred_proba = df['roberta_large_openai_detector_probability'] # Predicted labels

# Define the columns to check
results_path = path+f"//convergence_per_group_disparity.txt"
if os.path.exists(results_path):
    os.remove(results_path)

for col in features_columns:
    initial_thresholds_col = initial_thresholds[initial_thresholds['column']==col]
    min_acc_dict = initial_thresholds_col.set_index('label')['min_accuracy'].to_dict()
    min_f1_dict = initial_thresholds_col.set_index('label')['min_f1'].to_dict()
    print('min_acc_dict:',min_acc_dict)
    print('min_f1_dict:',min_f1_dict)
    unique_values_group = np.unique(initial_thresholds_col['group'])
    group_column = f'{unique_values_group[0]}-{col}'
    
    with open(results_path, "a") as f:
        
        f.write(f'{"=" * 70}\nGroup: {group_column}.\n{"-" * len(group_column)}')
        f.write(f'\nTolerance: {tolerance}; Penalty: {penalty}; Max. number of iterations: {max_iterations}; {frac*100} % of the dataset.')
        f.write(f'\nMin ACC threshold: {min_acc_dict}')
        f.write(f'\nMin F1 threshold: {min_f1_dict}\n\n')
    
    print("\n"+"|"*100+"\n")
    group_indices = {}
    for value in initial_thresholds_col['label']:
        #print (value)
        group_indices[value] = (df[col]==value)
        #print((df[col]==value))
        
    for acceptable_disparity in acceptable_disparities:
        print("\n"+"="*100)
        print(f"Optimized Thresholds for acceptable_disparity = {acceptable_disparity}:\n")

        optimized_thresholds, iterations, opt_learning_rate, opt_acc_dict, opt_f1_dict, is_convergence, delta = optimize_thresholds(
                                                y_true = y_true,
                                                y_pred_proba = y_pred_proba,
                                                groups = initial_thresholds_col['label'],
                                                initial_thresholds = initial_thresholds_values,
                                                group_indices = group_indices,
                                                acceptable_disparity = acceptable_disparity,
                                                max_iterations = max_iterations,
                                                learning_rate = learning_rate, 
                                                tolerance = tolerance, 
                                                penalty = penalty,
                                                min_acc_threshold = min_acc_dict,
                                                min_f1_threshold = min_f1_dict,
                                                group_column = group_column
                                            )
        
        #if iterations == max_iterations or :
        #    is_convergence = False
        #else:
        #    is_convergence = True
        
        logout_convergence = (
            f"Acceptable disparity: {acceptable_disparity}; "
            f"Convergence: {is_convergence}; "
            f'Iterations: {iterations}; '
            f'delta: {delta}; '
            f'Optimized Learning Rate: {opt_learning_rate}; '
            f'\nOptimized Accuracy: {opt_acc_dict}; '
            f'\nOptimized F1: {opt_f1_dict}.'
        )

        print(logout_convergence+'\n')
        with open(results_path, "a") as f:
            f.write(logout_convergence+'\n\n')
            
        for item in optimized_thresholds:
            print(f"Group: {item['group']}, Threshold: {item['threshold']:.7f}")
        
        if is_convergence == False: 
            break