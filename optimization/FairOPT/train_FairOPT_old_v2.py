import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import FairOPT_old_v2 as FairOPT

path = 'C://Users//Cynthia//Documents//MIT//datasets'
train_path = path+'//train_features.csv'

train_dataset = pd.read_csv(train_path)

#split by train and tesxt <- change this later
df = train_dataset.sample(frac=1, random_state=42)

# Length-based groups
length_groups = pd.cut(
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

# Sentiment and personality groups (ensure no NaN)
sentiment_groups = df['sentiment_label'].fillna('neutral').astype(str).values
personality_groups = df['personality'].fillna('unknown').astype(str).values

# Combine groups into a single group label
groups = pd.Series([
    f"{length}_{formality}_{sentiment}_{personality}"
    for length, formality, sentiment, personality in zip(length_groups, formality_groups, sentiment_groups, personality_groups)
]).values


# Prepare true labels and predicted probabilities
y_true = df['AI_written'].apply(lambda x: 1 if x == 'AI' else 0).values  # True labels
y_pred_proba = df['roberta_large_openai_detector_probability'].values     # Predicted probabilities the probability is learned from one model

# Initial thresholds (set to 0.5 for all groups)
initial_thresholds = {group: 0.5 for group in np.unique(groups)}


def optimize_thresholds(y_true, y_pred_proba, groups, initial_thresholds, acceptable_disparity, max_iterations, tolerance):
    # Create an instance of ThresholdOptimizer
    optimizer = FairOPT.ThresholdOptimizer(
        y_true = y_true,
        y_pred_proba = y_pred_proba,
        groups = groups,
        initial_thresholds = initial_thresholds,
        learning_rate=10**-2,
        max_iterations=max_iterations,
        acceptable_disparity=acceptable_disparity,  # Adjust based on your fairness criteria
        min_acc_threshold=0.5,  # Set realistic minimum accuracy
        min_f1_threshold=0.5,   # Set realistic minimum F1 score
        tolerance=tolerance,    # Decrease tolerance for stricter convergence criteria
        penalty=20  # Increase penalty to enforce stricter updates
    )

    # Optimize thresholds using gradient-based method
    thresholds, history, iteration = optimizer.optimize()
    
    # Save thresholds to a file
    file_path = path+f"//thresholds_disparity_{str(acceptable_disparity).replace('.', '_')}.txt"
    with open(file_path, 'w') as file:
        for group, threshold in thresholds.items():
            file.write(f"Group: {group}, Threshold: {threshold:.7f}\n")
    print("The optimized thresholds have been saved to:", file_path)

    # Move the results to the list
    optimized_thresholds_list = []
    for group, threshold in thresholds.items():
        optimized_thresholds_list.append({'group': group, 'threshold': threshold})

    return optimized_thresholds_list, iteration


acceptable_disparities =  [1, 0.2, 0.1, 0.01, 0.001]
max_iterations = 10**1
tolerance = 1e-4 #10**-5

results_path = path+f"//convergence_per_disparity.txt"
if os.path.exists(results_path):
    os.remove(results_path)
        
for acceptable_disparity in acceptable_disparities:
    print("\n"+"="*100)
    print(f"Optimized Thresholds for acceptable_disparity = {acceptable_disparity}:\n")
    optimized_thresholds, iterations = optimize_thresholds(y_true, y_pred_proba, groups, initial_thresholds, acceptable_disparity, max_iterations, tolerance)
    
    if iterations == max_iterations:
        is_convergence = False
    else:
        is_convergence = True
    
    logout_convergence = f"Acceptable disparity: {acceptable_disparity}, Convergence: {is_convergence}, Number of iterations: {iterations}, Maximum iterations: {max_iterations}, Tolerance: {tolerance}"
    print(logout_convergence+'\n')
    with open(results_path, "a") as f:        
        f.write(logout_convergence+'\n\n')
        
    #print("\n")
    for item in optimized_thresholds:
        print(f"Group: {item['group']}, Threshold: {item['threshold']:.7f}")