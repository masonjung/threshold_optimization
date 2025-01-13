import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t4_features.csv")

df.columns
df.shape

# Combine probabilities from all detectors
probability_columns = [
    # 'roberta_base_openai_detector_probability',
    # 'roberta_large_openai_detector_probability',
    'radar_probability'
]
df['combined_probability'] = df[probability_columns].mean(axis=1)

# Focus on combined probability between 0.00 and 1.00
df_filtered = df[(df['combined_probability'] >= 0.00) & (df['combined_probability'] <= 1.00)]

# Categorize text length
conditions = [
    (df_filtered['text_length'] < 1000),
    (df_filtered['text_length'] > 2500)
]
choices = ['short', 'long']
df_filtered['text_length_category'] = np.select(conditions, choices, default='medium')

# Thresholds for each personality and text length
thresholds = {
    'short': {
        'extroversion': 0.5074398,
        'neuroticism': 0.5134435,
        'agreeableness': 0.5000000,
        'conscientiousness': 0.4994540,
        'openness': 0.5015122,
    },
    'medium': {
        'extroversion': 0.4923216,
        'neuroticism': 0.4901528,
        'agreeableness': 0.4984677,
        'conscientiousness': 0.4898049,
        'openness': 0.4928704,
    },
    'long': {
        'extroversion': 0.5000000,
        'neuroticism': 0.5000000,
        'agreeableness': 0.5053591,
        'conscientiousness': 0.5125101,
        'openness': 0.5056818,
    }
}


# Colors for each personality (with higher contrast)
colors = {
    'extroversion': '#ff7f0e',  # Orange
    'neuroticism': '#1f77b4',  # Blue
    'agreeableness': '#2ca02c',  # Green
    'conscientiousness': '#d62728',  # Red
    'openness': '#9467bd',  # Purple
}

# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)

# Text length categories
text_length_categories = ['short', 'medium', 'long']

# Combined legend
handles, labels = [], []

for i, text_length in enumerate(text_length_categories):
    ax = axes[i]
    df_subset = df_filtered[df_filtered['text_length_category'] == text_length]
    
    # Plot histogram (bar chart) for each personality
    for personality, color in colors.items():
        df_personality = df_subset[df_subset['personality'] == personality]
        sns.histplot(
            data=df_personality,
            x='combined_probability',
            bins=15,
            ax=ax,
            color=color,
            alpha=0.1,
            label=f'{personality} (hist)',
            stat='percent'
        )
        
        # Plot KDE for each personality
        kde_plot = sns.kdeplot(
            data=df_personality,
            x='combined_probability',
            ax=ax,
            linewidth=2,
            color=color,
            label=f'{personality} (KDE)',
            bw_adjust=0.1,
            clip=(0, 1)
        )
        
        # Add threshold line for this text length and personality
        threshold = thresholds[text_length][personality]
        threshold_line = ax.axvline(
            x=threshold,
            color=color,
            linestyle='--',
            linewidth=1.5,
            label=f'{personality} threshold'
        )
        if i == 0:  # Only collect legend items once
            handles.append(kde_plot.get_lines()[0])
            labels.append(f'{personality} (KDE)')
            handles.append(threshold_line)
            labels.append(f'{personality} threshold')
    
    # Add universal static threshold line
    static_line = ax.axvline(
        x=0.5,
        color='black',
        linestyle='-',
        linewidth=3,
        label='Static Threshold'
    )
    # Add universal AUROC threshold line
    auroc_line = ax.axvline(
        x=0.9984415,
        color='gray',  # Dark gray for higher contrast
        linestyle='-',
        linewidth=3,  # Increased line width
        label='AUROC Threshold'
)

    if i == 0:  # Only collect legend items once
        handles.extend([static_line, auroc_line])
        labels.extend(['Static Threshold', 'AUROC Threshold'])

    # Set title and labels
    ax.set_title(f'{text_length.capitalize()} Text', fontsize=14, loc='left')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Detector Probability', fontsize=12)
    if i == 0:
        ax.set_ylabel('Density (%)', fontsize=12)

# Add a combined legend outside the plot
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fontsize=10)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
