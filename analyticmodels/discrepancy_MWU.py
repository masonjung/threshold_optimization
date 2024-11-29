import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import mannwhitneyu, ks_2samp

class FairThresholdAnalysis:
    def __init__(self, filepath, feature_columns, probability_columns, quantile_range=(0.25, 0.75)):
        # Load dataset from the given file path
        self.df = pd.read_csv(filepath)
        self.feature_columns = feature_columns
        self.probability_columns = probability_columns
        self.quantile_range = quantile_range
        self.results = []

    def calculate_discrepancies(self):
        # Iterate through each probability column
        for prob_col in self.probability_columns:
            # Ensure the probability values are between 0 and 1
            self.df[prob_col] = self.df[prob_col].clip(0, 1)

            # Filter dataset to focus on the "gray zone" (probability values between quantile_range)
            lower_quantile, upper_quantile = self.quantile_range
            df_filtered = self.df[
                (self.df[prob_col] > lower_quantile) & 
                (self.df[prob_col] < upper_quantile)
            ]

            # Group by the feature columns
            grouped = df_filtered.groupby(self.feature_columns)
            group_keys = list(grouped.groups.keys())

            # Iterate over all unique pairs of group keys
            for val1, val2 in itertools.combinations(group_keys, 2):
                group1 = grouped.get_group(val1)[prob_col]
                group2 = grouped.get_group(val2)[prob_col]

                # Ensure groups have sufficient data
                if len(group1) < 5 or len(group2) < 5:
                    continue

                # Perform the Mann-Whitney U test
                stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

                # Calculate the median values
                group1_median = group1.median()
                group2_median = group2.median()
                diff = abs(group1_median - group2_median)

                # Store the results if p-value < 0.05
                if p_value < 0.05:
                    self.results.append({
                        'probability_column': prob_col,
                        'combination': (val1, val2),
                        'median_difference': diff,
                        'p_value': p_value
                    })

        # Sort results by median difference to determine the biggest, second biggest, and lowest discrepancies
        self.results.sort(key=lambda x: x['median_difference'], reverse=True)



    def plot_individual_kde(self, quantile_range=(0.25, 0.75)):
        # Create a figure with subplots for individual KDE plots
        fig, axes = plt.subplots(1, 4, figsize=(32, 8), sharey=True)  # Updated to 4 subplots
        colors = ['#d62728', '#e74c3c', '#90ee90', '#2ca02c']  # Strong red, red-orange, light green, green
        contrast_colors = ['#ff4c4c', '#ff6347', '#a3e4a3', '#3cb371']  # Contrasting shades for each group

        labels = ['biggest', 'second biggest', 'second lowest', 'lowest']  # Updated labels

        for idx, result in enumerate(self.results[:4]):
            # Extract information from the result
            prob_col = result['probability_column']
            val1, val2 = result['combination']
            group1_info = dict(zip(self.feature_columns, val1))
            group2_info = dict(zip(self.feature_columns, val2))

            # Prepare data for plotting
            df_subset1 = self.df[(self.df[self.feature_columns] == val1).all(axis=1)].copy()
            df_subset2 = self.df[(self.df[self.feature_columns] == val2).all(axis=1)].copy()
            df_subset1['Subgroup'] = 'Group 1'
            df_subset2['Subgroup'] = 'Group 2'
            df_combined = pd.concat([df_subset1, df_subset2])

            # Filter the combined dataset to focus on the "gray zone" (probability values between quantile_range)
            lower_quantile, upper_quantile = quantile_range
            df_combined_gray_zone = df_combined[
                (df_combined[prob_col] >= lower_quantile) & 
                (df_combined[prob_col] <= upper_quantile)
            ]

            # Clip the probability values to be within 0 and 1
            df_combined_gray_zone[prob_col] = df_combined_gray_zone[prob_col].clip(0, 1)

            # Separate the filtered data into the two groups
            group1_data_gray_zone = df_combined_gray_zone[df_combined_gray_zone['Subgroup'] == 'Group 1'][prob_col]
            group2_data_gray_zone = df_combined_gray_zone[df_combined_gray_zone['Subgroup'] == 'Group 2'][prob_col]

            # Check if there is enough data for plotting
            if len(group1_data_gray_zone) < 5 or len(group2_data_gray_zone) < 5:
                print(f"Not enough data points for meaningful plot in {labels[idx]} discrepancy.")
                continue

            # Plotting KDE for both groups with contrasting colors
            sns.kdeplot(group1_data_gray_zone, label='Group 1', color=colors[idx], linestyle='-', fill=True, alpha=0.8, bw_adjust=0.9, ax=axes[idx])
            sns.kdeplot(group2_data_gray_zone, label='Group 2', color=contrast_colors[idx], linestyle='--', fill=True, alpha=0.8, bw_adjust=0.9, ax=axes[idx])
            axes[idx].set_title(f'{labels[idx].capitalize()} Discrepancy', fontsize=18, weight='bold')
            axes[idx].set_xlabel(f'Probability Score ({quantile_range[0]} - {quantile_range[1]})', fontsize=14)
            axes[idx].set_xlim(lower_quantile, upper_quantile)  # Set x-axis limit to the quantile range
            axes[idx].text(0.5, 1.1, f'Group 1: {group1_info}\nGroup 2: {group2_info}', ha='center', va='top', transform=axes[idx].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))
            if idx == 0:
                axes[idx].set_ylabel('Density', fontsize=14)
            axes[idx].legend(fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle('KDE Plots for Biggest, Second Biggest, Second Lowest, and Lowest Discrepancies', fontsize=22, weight='bold')
        plt.show()



    def analyze_and_plot(self, quantile_ranges=[(0.25, 0.75)]):
        # Calculate discrepancies
        self.calculate_discrepancies()

        # Plot individual KDEs for each quantile range
        for quantile_range in quantile_ranges:
            print(f"Visualizing for quantile range: {quantile_range}")
            self.plot_individual_kde(quantile_range)


# Usage example
if __name__ == "__main__":
    filepath = r"C:\Users\minse\Desktop\Programming\FairThresholdOptimization\datasets\train_features.csv"  # Adjust the file path accordingly
    feature_columns = ['personality', 'sentiment_label', "formality_label", "length_label"]
    probability_columns = [
        'roberta_base_openai_detector_probability',
        'roberta_large_openai_detector_probability',
        'radar_probability'
    ]

    analysis = FairThresholdAnalysis(filepath, feature_columns, probability_columns)
    analysis.analyze_and_plot(quantile_ranges=[(0.01, 0.99), (0.1, 0.9)])