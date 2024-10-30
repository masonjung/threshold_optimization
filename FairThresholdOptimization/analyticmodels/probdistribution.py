import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE_d3.csv")

# Select relevant columns
probability_columns = [
    'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    'radar_probability'
]

# Remove the top 1% longest texts
df_filtered = df[df['num_chars'] <= df['num_chars'].quantile(0.99)]

# Set up the plotting area
plt.figure(figsize=(15, 12))

# Loop through each classifier and create a scatter plot
for i, col in enumerate(probability_columns, 1):
    plt.subplot(3, 1, i)
    sns.scatterplot(x=df_filtered['num_chars'], y=df_filtered[col], color=df_filtered['AI_written'].map({0: 'blue', 1: 'red'}), label=col)
    plt.xlabel('Text Length (Number of Characters)')
    plt.ylabel('Probability')
    plt.title(f'Text Length vs. {col}')
    plt.legend()

plt.tight_layout()
plt.show()
