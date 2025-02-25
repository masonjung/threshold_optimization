import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = [PATH]
df.columns
df.shape


# Combine probabilities from all detectors
probability_columns = [
    # 'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    # 'radar_probability'
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

# Set up the figure for histograms and density estimation for AI-written and human-written texts
fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

# Plot histogram and KDE for combined probability by text length category for AI-written and human-written texts
for i, ai_written in enumerate([1, 0]):
    ax = axes[i]
    for length_category, color in zip(['short', 'medium', 'long'], ['red', 'green', 'blue']):
        # Filter by AI_written and text length category
        df_subset = df_filtered[(df_filtered['AI_written'] == ai_written) & (df_filtered['text_length_category'] == length_category)]
        label = f"{length_category.capitalize()} text"
        sns.histplot(df_subset['combined_probability'], ax=ax, kde=False, bins=25, color=color, alpha=0.1, label=label, stat='percent')
        sns.kdeplot(df_subset['combined_probability'], ax=ax, color=color, linewidth=2, linestyle='-', bw_adjust=0.02, clip=(0, 1))
        # sns.kdeplot(df_subset['combined_probability'], ax=ax, color=color, linewidth=2, linestyle='-', label=f"{label} (Smoothed)", bw_adjust=0.02, clip=(0, 1))
    # Force x-axis limits
    ax.set_xlim(0, 1)
    # Draw a static threshold line
    threshold = 0.5
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=3, label='Static threshold (0.5)')

    # # Draw adaptive threshold lines for each group
    # adaptive_thresholds = {
    #     'short': df_filtered[df_filtered['text_length_category'] == 'short']['combined_probability'].mean(),
    #     'medium': df_filtered[df_filtered['text_length_category'] == 'medium']['combined_probability'].mean(),
    #     'long': df_filtered[df_filtered['text_length_category'] == 'long']['combined_probability'].mean()
    # }
    # for length_category, linestyle, color in zip(['short', 'medium', 'long'], [':', '-.', '--'], ['red', 'green', 'blue']):
    #     adaptive_threshold = adaptive_thresholds[length_category]
    #     ax.axvline(x=adaptive_threshold, color=color, linestyle=linestyle, linewidth=2, label=f'Adaptive threshold for {length_category} text')
    #     # ax.axvline(x=adaptive_threshold, color=color, linestyle=linestyle, linewidth=2, label=f'Adaptive Threshold for ({length_category.capitalize()}) = {adaptive_threshold:.4f}')


    # Set labels and title for each subplot
    ax.set_ylabel('Percentage (%)', size = 25)
    title = 'AI-written text' if ai_written == 1 else 'Human-written text'
    # ax.set_title(f'{title}')

    # Adjust text position to the left upper side of the box
    ax.text(0.7, 0.95, title, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=25, bbox=dict(facecolor='white', alpha=0.0))
    ax.legend(loc='upper left', fontsize=20)
    ax.tick_params(axis='y', labelsize=20)
    if i == 1:
        ax.set_yticks(ax.get_yticks()[1:]) # eliminate the 0 of y-axis because x-axis provides 0
# Set common x-axis label
plt.xlabel('AI text classifier probability by length', size = 25)
plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=20)

# Adjust layout and show the plots
plt.tight_layout()
plt.show()
