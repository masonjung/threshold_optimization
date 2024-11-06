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
    # 'radar_probability'
]

# Remove the top 1% longest texts
df_filtered = df[df['num_chars'] <= df['num_chars'].quantile(0.99)]

# Loop through each classifier and AI_written category
for col in probability_columns:
    for ai_written in [0, 1]:
        # Filter by AI_written value
        df_subset = df_filtered[df_filtered['AI_written'] == ai_written]
        
        # Set up the figure
        plt.figure(figsize=(10, 6))
        
        # Plot the scatter plot
        sns.scatterplot(x=df_subset['num_chars'], y=df_subset[col], hue=df_subset['AI_written'], palette={0: 'blue', 1: 'red'})
        plt.xlabel('Text Length (Number of Characters)')
        plt.ylabel('Probability')
        plt.title(f'Text Length vs. {col} (AI_written = {ai_written})')
        plt.legend(title='AI Written', labels=['Human Written', 'AI Written'])
        
        # Show the plot
        plt.tight_layout()
        plt.show()
