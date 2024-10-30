import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# Load the dataset
df = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE_d3.csv")

# Analyze differences between two specific subgroups based on probabilities from detectors
def calculate_greatest_difference(df, feature_columns, probability_column):
    max_diff = 0
    best_combination = None

    # Iterate over all combinations of feature values
    for val1 in df.groupby(feature_columns).groups.keys():
        for val2 in df.groupby(feature_columns).groups.keys():
            if val1 != val2:
                group1 = df[(df[list(feature_columns)] == val1).all(axis=1)][probability_column]
                group2 = df[(df[list(feature_columns)] == val2).all(axis=1)][probability_column]
                diff = abs(group1.mean() - group2.mean())
                if diff > max_diff:
                    max_diff = diff
                    best_combination = (val1, val2)

    return best_combination, max_diff

# List of features and probabilities to analyze
features_to_analyze = ['educational_level', 'sentiment_label'] # AI written should not be included because it necessarily shows the discrepancy
probability_columns = [
    'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    'radar_probability'
]

# Find the greatest difference between subgroups for each probability column
greatest_differences = []
for prob_col in probability_columns:
    best_combination, max_diff = calculate_greatest_difference(df, features_to_analyze, prob_col)
    greatest_differences.append((prob_col, best_combination, max_diff))

# Set up the plotting area
plt.figure(figsize=(12, 8))

# Plot the greatest difference for each probability column
for i, (prob_col, best_combination, max_diff) in enumerate(greatest_differences, 1):
    group1, group2 = best_combination
    df_subset1 = df[(df[list(features_to_analyze)] == group1).all(axis=1)]
    df_subset2 = df[(df[list(features_to_analyze)] == group2).all(axis=1)]
    df_subset1['Subgroup'] = 'Group 1'
    df_subset2['Subgroup'] = 'Group 2'
    df_combined = pd.concat([df_subset1, df_subset2])

    plt.subplot(len(probability_columns), 1, i)
    sns.boxplot(x='Subgroup', y=prob_col, data=df_combined, palette='viridis')
    plt.xlabel('Subgroup')
    plt.ylabel('Probability')
    plt.title(f'Greatest Discrepancy in {prob_col} (Subgroups: {group1} vs {group2}) - Difference: {max_diff:.2f}')

plt.tight_layout()
plt.show()
