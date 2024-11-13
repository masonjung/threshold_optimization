import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.stats import mannwhitneyu

# Load the dataset
df = pd.read_csv(r"C:\Users\minse\Desktop\Programming\FairThresholdOptimization\datasets\Training_dataset\Train_RAID_MAGE_d3.csv")

def calculate_greatest_difference(df, feature_columns, probability_column, quantile_range=(0.25, 0.75)):
    # Define the quantile limits
    lower_quantile, upper_quantile = quantile_range
    quantile_limits = df[probability_column].quantile([lower_quantile, upper_quantile])

    # Filter dataset to only include rows within the quantile range of the probability column
    df_filtered = df[
        (df[probability_column] >= quantile_limits.iloc[0]) &
        (df[probability_column] <= quantile_limits.iloc[1])
    ]

    # Group the filtered DataFrame once
    grouped = df_filtered.groupby(feature_columns)
    group_keys = list(grouped.groups.keys())

    max_diff = 0
    best_combination = None
    min_p_value = 1  # Initialize with the highest possible p-value

    # Iterate over all unique pairs of group keys
    for val1, val2 in itertools.combinations(group_keys, 2):
        group1 = grouped.get_group(val1)[probability_column]
        group2 = grouped.get_group(val2)[probability_column]

        # Ensure groups have sufficient data
        if len(group1) < 5 or len(group2) < 5:
            continue

        # Perform the Mann-Whitney U test
        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        # Calculate the median values
        group1_median = group1.median()
        group2_median = group2.median()
        diff = abs(group1_median - group2_median)

        # Update if difference is greater
        if diff > max_diff:
            max_diff = diff
            best_combination = (val1, val2)
            min_p_value = p_value

    return best_combination, max_diff, min_p_value

# List of features and probabilities to analyze
features_to_analyze = ['educational_level', 'sentiment_label']
probability_columns = [
    'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    'radar_probability'
]

# Define the quantile range for probability values
quantile_range = (0.25, 0.75)  # Adjust these values as needed

greatest_differences = []
best_probability_column = None
max_diff_overall = 0
best_combination_overall = None
min_p_value_overall = 1

for prob_col in probability_columns:
    best_combination, max_diff, min_p_value = calculate_greatest_difference(
        df,
        features_to_analyze,
        prob_col,
        quantile_range
    )
    greatest_differences.append((prob_col, best_combination, max_diff, min_p_value))
    if max_diff > max_diff_overall:
        max_diff_overall = max_diff
        best_probability_column = prob_col
        best_combination_overall = best_combination
        min_p_value_overall = min_p_value

# Extract information about the two groups
val1, val2 = best_combination_overall
group1_info = dict(zip(features_to_analyze, val1))
group2_info = dict(zip(features_to_analyze, val2))

print("Group 1 characteristics:", group1_info)
print("Group 2 characteristics:", group2_info)

# Prepare data for plotting
df_subset1 = df[
    (df[features_to_analyze] == val1).all(axis=1)
].copy()
df_subset2 = df[
    (df[features_to_analyze] == val2).all(axis=1)
].copy()
df_subset1['Subgroup'] = 'Group 1'
df_subset2['Subgroup'] = 'Group 2'
df_combined = pd.concat([df_subset1, df_subset2])

# Filter the combined dataset to focus on the quantile range for visualization
quantile_limits_combined = df_combined[best_probability_column].quantile([0.25, 0.75])
df_combined = df_combined[
    (df_combined[best_probability_column] >= quantile_limits_combined.iloc[0]) &
    (df_combined[best_probability_column] <= quantile_limits_combined.iloc[1])
]

# Round the p-value to two decimal places
rounded_p_value = round(min_p_value_overall, 2)

# Alternatively, represent the p-value based on thresholds
if min_p_value_overall < 0.01:
    p_value_text = "p < 0.01"
elif min_p_value_overall < 0.05:
    p_value_text = "p < 0.05"
else:
    p_value_text = f"p = {rounded_p_value}"

# Plotting with improved colors
plt.figure(figsize=(12, 8))

custom_palette = {'Group 1': '#1f77b4', 'Group 2': '#ff7f0e'}  # Custom colors

sns.violinplot(
    x='Subgroup',
    y=best_probability_column,
    data=df_combined,
    palette=custom_palette,
    inner='quartile',
    cut=0
)

sns.stripplot(
    x='Subgroup',
    y=best_probability_column,
    data=df_combined,
    color='blue',
    alpha=0.5,
    jitter=True
)

# skip displaying x and y labels
plt.xlabel('') # plt.xlabel('Subgroup', fontsize=14)
plt.ylabel('') # plt.ylabel('Probability', fontsize=14)

# Detailed title including group characteristics and formatted p-value
plt.title(
    f'Greatest Discrepancy in {best_probability_column}\n'
    f'Group 1 ({group1_info}) vs Group 2 ({group2_info})\n'
    f'Median Difference: {max_diff_overall:.2f}, {p_value_text}',
    fontsize=16
)

plt.ylim(0, 1)
plt.grid(axis='y', linestyle='-', alpha=0.7)
plt.tight_layout()
plt.show()
