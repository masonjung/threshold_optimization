import pandas as pd
import numpy as np
import FairOPT
import acc_f1_baseline
import time
start_time = time.time()

# Minseok's path
#path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
#train_path = path+"\\train_features.csv"

# Cyntia's path
path = 'C://Users//Cynthia//Documents//IpParis_C//MIT//datasets'
train_path = path+'//train_features.csv'



##########################################
# Hyperparameters
##########################################

acceptable_disparity = 0.05 #[1.00, 0.30, 0.25, 0.21, 0.20, 0.19, 0.15, 0.10]
print(f'acceptable_disparity: {acceptable_disparity}')

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

acc_f1_baseline_df = baseline_df.groupby(['group', 'column', 'label']).agg(
        min_accuracy=('static_accuracy', lambda x: min(x.min(), baseline_df.loc[x.index, 'roc_accuracy'].min())),
        min_f1=('static_f1', lambda x: min(x.min(), baseline_df.loc[x.index, 'roc_f1'].min()))
    ).reset_index()


#print(df.shape)
#print(acc_f1_baseline_df)



##########################################
# Optimize thresholds
##########################################

# True labels and Predicted probabilities learned from one model
y_true = df['AI_written']  # True labels
y_pred_proba = df['roberta_large_openai_detector_probability'] # Predicted labels
print(df.columns)

for _, col in enumerate(features_columns):
    acc_f1_col = acc_f1_baseline_df[acc_f1_baseline_df['column']==col]
    print(f'{acc_f1_col}')
    min_acc_dict = acc_f1_col.set_index('label')['min_accuracy'].to_dict()
    min_f1_dict = acc_f1_col.set_index('label')['min_f1'].to_dict()
    unique_values_group = np.unique(acc_f1_baseline_df['group'])
    
    group_indices = {}
    for value in acc_f1_col['label']:
        group_indices[value] = (df[col]==value)
    
    optimizer = FairOPT.ThresholdOptimizer(
        y_true = y_true,
        y_pred_proba = y_pred_proba,
        
        group_indices = group_indices,
        min_acc = min_acc_dict,
        min_f1 = min_f1_dict,
        min_disparity = acceptable_disparity,
                
        learning_rate = learning_rate,
        penalty = penalty,
        max_iteration_minimize = max_iteration_minimize
    )

    # Optimize thresholds using gradient-based method
    is_convergence = optimizer.optimize()
    if is_convergence:
        break
    print(f'Convergence: {is_convergence}')
    
print(f'Convergence: {is_convergence}')
end_time = time.time()
execution_time = end_time - start_time
minutes = execution_time // 60
seconds = execution_time % 60
print(f"The code took {minutes} minutes and {seconds:.2f} seconds to run.")