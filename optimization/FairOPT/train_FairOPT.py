import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from itertools import combinations
from itertools import product
import FairOPT

# Minseok's path
#path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets"
#train_path = path+"\\train_features.csv"


# Cyntia's path
path = 'C://Users//Cynthia//Documents///MIT//datasets'
train_path = path+'//train_features.csv'



# hyperparameters
acceptable_disparities =  [1, 0.5, 0.2] #[1, 0.5, 0.25, 0.24, 0.239, 0.238, 0.237, 0.236, 0.235, 0.234, 0.233, 0.232, 0.231, 0.23, 0.229, 0.228, 0.227, 0.226, 0.225, 0.223, 0.22, 0.21, 0.2] # [1, 0.5, 0.2, 0.1, 0.01, 0.001]
max_iterations = 2*10**3
learning_rate = 10**-3
tolerance = 1e-3 #10**-5
min_acc_threshold = 0.5 #0.5
min_f1_threshold = 0.5 #0.5
num_features = [1] #[1, 2, 3, 4]



train_dataset = pd.read_csv(train_path)
train_dataset.shape


#split by train and tesxt <- change this later
#df = train_dataset.sample(frac=0.1, random_state=42)
frac = 1.0 # sample fraction
df = train_dataset.groupby(['AI_written'], group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=42))
df.reset_index(drop=True, inplace=True)


# Length-based groups
length_groups = pd.cut(
    df['text_length'].dropna(),
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values

df['length_feature'] = pd.cut(
    df['text_length'].dropna(),
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values


# Formality-based groups
formality_groups = pd.cut(
    df['formality'].dropna(),
    bins=[0, 50, np.inf],
    labels=['informal', 'formal']
).astype(str).values
formality_groups = [x for x in formality_groups if x != 'nan']

df['formality_feature'] = pd.cut(
    df['formality'].dropna(),
    bins=[0, 50, np.inf],
    labels=['informal', 'formal']
).astype(str).values


# Sentiment and personality groups (ensure no NaN)
sentiment_groups = df['sentiment_label'].fillna('neutral').astype(str).values
personality_groups = df['personality'].fillna('unknown').astype(str).values

# Prepare true labels and predicted probabilities
y_true = df['AI_written']  # True labels
y_pred_proba = df['roberta_large_openai_detector_probability']     # Predicted probabilities the probability is learned from one model


def generate_groups_and_thresholds(length_groups, formality_groups, sentiment_groups, personality_groups, selected_features):
    """
    Combine groups into labels and generate initial thresholds for each combination.

    Parameters:
    length_groups (list): Groups based on length.
    formality_groups (list): Groups based on formality.
    sentiment_groups (list): Groups based on sentiment.
    personality_groups (list): Groups based on personality.
    selected_features (list): List of feature groups to include (e.g., [length_groups, formality_groups]).

    Returns:
    dict: Dictionary with keys 'groups' and 'initial_thresholds'.
    """
    # Fixed feature names
    all_features = {
        'length': length_groups,
        'formality': formality_groups,
        'sentiment': sentiment_groups,
        'personality': personality_groups
    }

    # Determine which features are selected
    feature_names = [name for name, group in all_features.items() if group in selected_features]
    feature_values = [all_features[name] for name in feature_names]

    # Generate all combinations of selected features
    combined_values = list(product(*feature_values))
    group_values = ["_".join(map(str, values)) for values in combined_values]

    # Create labels for the combined group
    groups_dict = pd.Series(group_values).values

    # Generate initial thresholds for all groups
    unique_groups = np.unique(groups_dict)
    initial_thresholds = {group: 0.5 for group in unique_groups}

    #return {
    #    'groups': groups_dict,
    #    'initial_thresholds': initial_thresholds
    #}
    return groups_dict, initial_thresholds



def optimize_thresholds(y_true, y_pred_proba, groups, initial_thresholds, group_indices, acceptable_disparity, max_iterations, tolerance, min_acc_threshold, min_f1_threshold,count_group):
    optimizer = FairOPT.ThresholdOptimizer(
        y_true = y_true,
        y_pred_proba = y_pred_proba,
        groups = groups,
        initial_thresholds = initial_thresholds,
        group_indices = group_indices,
        learning_rate=learning_rate,
        max_iterations=max_iterations,
        acceptable_disparity=acceptable_disparity,  # Adjust based on your fairness criteria
        min_acc_threshold=min_acc_threshold,  # Set realistic minimum accuracy
        min_f1_threshold=min_f1_threshold,   # Set realistic minimum F1 score
        tolerance=tolerance,    # Decrease tolerance for stricter convergence criteria
        penalty=20  # Increase penalty to enforce stricter updates
    )

    # Optimize thresholds using gradient-based method
    #thresholds, history, iteration = optimizer.optimize()
    thresholds, iteration = optimizer.optimize()
    
    # Save thresholds to a file
    file_path = path+f"//thresholds_group_{str(count_group).zfill(2)}_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
    with open(file_path, 'w') as file:
        for group, threshold in thresholds.items():
            file.write(f"Group: {group}, Threshold: {threshold:.7f}\n")
    print("The optimized thresholds have been saved to:", file_path)

    # Move the results to the list
    optimized_thresholds_list = []
    for group, threshold in thresholds.items():
        optimized_thresholds_list.append({'group': group, 'threshold': threshold})

    return optimized_thresholds_list, iteration


uni_length_groups = np.unique(length_groups).tolist()
uni_formality_groups = np.unique(formality_groups).tolist()
uni_sentiment_groups = np.unique(sentiment_groups).tolist()
uni_personality_groups = np.unique(personality_groups).tolist()

features = [uni_length_groups, uni_formality_groups, uni_sentiment_groups, uni_personality_groups]
feature_combinations = [list(combo) for r in num_features for combo in combinations(features, r)]

# Define the columns to check
columns_to_check = ['length_feature','formality_feature', 'sentiment_label', 'personality'] #
results_path = path+f"//convergence_per_group_disparity.txt"
#group_labels_path = path+f"//group_labels.txt"
if os.path.exists(results_path):
    os.remove(results_path)

count_group = 0
for selected_features in feature_combinations:
    count_group += 1
    with open(results_path, "a") as f:
        f.write(f'{"=" * 70}\nGroup {str(count_group).zfill(2)}: {selected_features}\n\n\n')
            
    print("\n"+"|"*100+"\n")
    groups, initial_thresholds = generate_groups_and_thresholds(uni_length_groups, uni_formality_groups, uni_sentiment_groups, uni_personality_groups, selected_features)
    
    group_indices = {}
    for group in groups:
        df['indices'] = True  # Initialize all values to False
        words = group.split('_')
        for word in words:
            # Check if the word is in any of the specified columns
            count_col = 0
            for column in columns_to_check:
                count_col += 1
                aux = (df[column] == word)                
                counts = (df[column] == word).value_counts()
                if counts.get(True, 0) > 0:
                    df['indices'] = df['indices'] & (df[column] == word)
                    break
        group_indices[group] = df['indices']
    
    for group, indices in group_indices.items():
        true_count = indices.sum() 
        print(f"Group: {group}, True count: {true_count:,}")

        
    for acceptable_disparity in acceptable_disparities:
        print("\n"+"="*100)
        print(f"Optimized Thresholds for acceptable_disparity = {acceptable_disparity}:\n")
        optimized_thresholds, iterations = optimize_thresholds(
                                                y_true=y_true,
                                                y_pred_proba=y_pred_proba,
                                                groups=groups,
                                                initial_thresholds=initial_thresholds,
                                                group_indices=group_indices,
                                                acceptable_disparity=acceptable_disparity,
                                                max_iterations=max_iterations,
                                                tolerance=tolerance,
                                                min_acc_threshold=min_acc_threshold,
                                                min_f1_threshold=min_f1_threshold,
                                                count_group=count_group
                                            )
        
        if iterations == max_iterations:
            is_convergence = False
        else:
            is_convergence = True
        
        logout_convergence = (
            f"Acceptable disparity: {acceptable_disparity}, "
            f"Convergence: {is_convergence}, "
            f"Number of iterations: {iterations}, "
            f"Maximum iterations: {max_iterations}, "
            f"Tolerance: {tolerance}, "
            f"Min ACC threshold: {min_acc_threshold}, "
            f"Min F1 threshold: {min_f1_threshold}."
        )

        print(logout_convergence+'\n')
        with open(results_path, "a") as f:
            f.write(logout_convergence+'\n\n')
            
        for item in optimized_thresholds:
            print(f"Group: {item['group']}, Threshold: {item['threshold']:.7f}")