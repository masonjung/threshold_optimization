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

# Focus on combined probability between 0 and 1
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
        'extroversion': 0.2515866,
        'neuroticism': 0.2501819,
        'agreeableness': 0.2525938,
        'conscientiousness': 0.2489418,
        'openness': 0.2497191,
    },
    'medium': {
        'extroversion': 0.2482100,
        'neuroticism': 0.2463024,
        'agreeableness': 0.2467731,
        'conscientiousness': 0.2492236,
        'openness': 0.2474614,
    },
    'long': {
        'extroversion': 0.2500000,
        'neuroticism': 0.2500000,
        'agreeableness': 0.2500000,
        'conscientiousness': 0.2567252,
        'openness': 0.2517045,
    }
}

# Colors for each personality
colors = {
    'extroversion': '#ff7f0e',  # Orange
    'neuroticism': '#1f77b4',  # Blue
    'agreeableness': '#2ca02c',  # Green
    'conscientiousness': '#d62728',  # Red
    'openness': '#9467bd',  # Purple
}

# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Text length categories
text_length_categories = ['short', 'medium', 'long']

# Combined legend
handles, labels = [], []

for i, text_length in enumerate(text_length_categories):
    ax = axes[i]
    df_subset = df_filtered[df_filtered['text_length_category'] == text_length]
    
    # Plot CDF for each personality
    for personality, color in colors.items():
        df_personality = df_subset[df_subset['personality'] == personality]
        sorted_probs = np.sort(df_personality['combined_probability'].dropna())
        cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        cdf_plot, = ax.plot(
            sorted_probs,  # X-axis is Detector Probability
            cdf,  # Y-axis is Cumulative Probability
            color=color,
            label=f'{personality} (CDF)',
            linewidth=2
        )
        
        # Add threshold line for this text length and personality
        threshold = thresholds[text_length][personality]
        threshold_line = ax.axvline(
            x=threshold,  # Threshold now on x-axis
            color=color,
            linestyle='-.',
            linewidth=1.5,
            label=f'{personality} threshold'
        )
        if i == 0:  # Only collect legend items once
            handles.append(cdf_plot)
            labels.append(f'probability of {personality}')
            handles.append(threshold_line)
            labels.append(f'threshold for {personality}')
    
    # Add universal static threshold line
    static_line = ax.axvline(
        x=0.5,  # Static threshold on x-axis
        color='black',
        linestyle='-.',
        linewidth=2,
        label='static threshold'
    )
    auroc_line = ax.axvline(
        x=0.9984415,  # AUROC threshold on x-axis
        color='gray',
        linestyle='-.',
        linewidth=2,
        label='ROCFPR threshold'
    )
    if i == 0:  # Only collect legend items once
        handles.extend([static_line, auroc_line])
        labels.extend(['Static threshold', 'ROCFPR threshold'])
        # Set the font size of x and y ticks
        ax.tick_params(axis='both', which='major', labelsize=16)
    # Set title and labels
    ax.set_title(f'{text_length.capitalize()} length text', fontsize=16, loc='left', pad=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Detector probability', fontsize=16)  # x-axis label
    if i == 0:
        ax.set_ylabel('Cumulative density', fontsize=16)  # y-axis label

# Add a combined legend outside the plot
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=16)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
