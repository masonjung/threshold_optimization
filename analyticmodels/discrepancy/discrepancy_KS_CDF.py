import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.stats import ks_2samp


class FairThresholdAnalysis:
    def __init__(self, filepath, feature_columns, probability_columns, quantile_range=(0.25, 0.75)):
        self.df = pd.read_csv(filepath)
        self.feature_columns = feature_columns
        self.probability_columns = probability_columns
        self.quantile_range = quantile_range
        self.results = []

    def calculate_discrepancies(self):
        for prob_col in self.probability_columns:
            self.df[prob_col] = self.df[prob_col].clip(0, 1)

            lower_quantile, upper_quantile = self.quantile_range
            df_filtered = self.df[
                (self.df[prob_col] > lower_quantile) &
                (self.df[prob_col] < upper_quantile)
            ]

            grouped = df_filtered.groupby(self.feature_columns)
            group_keys = list(grouped.groups.keys())

            for val1, val2 in itertools.combinations(group_keys, 2):
                group1 = grouped.get_group(val1)[prob_col]
                group2 = grouped.get_group(val2)[prob_col]

                if len(group1) < 5 or len(group2) < 5:
                    continue

                ks_stat, p_value = ks_2samp(group1, group2)

                if p_value < 0.05:
                    self.results.append({
                        'probability_column': prob_col,
                        'combination': (val1, val2),
                        'ks_statistic': ks_stat,
                        'p_value': p_value
                    })

        self.results.sort(key=lambda x: x['ks_statistic'], reverse=True)

    def print_discrepancy_details(self):
        print("Detailed Discrepancy Information:")
        top_2 = self.results[:2]
        bottom_2 = self.results[-2:]
        selected_results = top_2 + bottom_2

        for idx, result in enumerate(selected_results):
            prob_col = result['probability_column']
            val1, val2 = result['combination']
            group1_info = dict(zip(self.feature_columns, val1))
            group2_info = dict(zip(self.feature_columns, val2))
            ks_stat = result['ks_statistic']
            p_value = result['p_value']

            print(f"  Group 1: {group1_info}")
            print(f"  Group 2: {group2_info}")
            print(f"  KS Statistic: {ks_stat:.4f}")
            print(f"  p-value: {p_value:.4e}")

    def plot_individual_kde(self, quantile_range=(0.25, 0.75), fontsize=12):
        top_2 = self.results[:2]
        bottom_2 = self.results[-2:]
        selected_results = top_2 + bottom_2

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)

        colors = ['red', 'orangered', 'seagreen', 'green']

        for idx, result in enumerate(selected_results):
            prob_col = result['probability_column']
            val1, val2 = result['combination']

            df_subset1 = self.df[(self.df[self.feature_columns] == val1).all(axis=1)].copy()
            df_subset2 = self.df[(self.df[self.feature_columns] == val2).all(axis=1)].copy()
            df_subset1['Subgroup'] = 'Group 1'
            df_subset2['Subgroup'] = 'Group 2'
            df_combined = pd.concat([df_subset1, df_subset2])

            lower_quantile, upper_quantile = quantile_range
            df_combined_gray_zone = df_combined[
                (df_combined[prob_col] >= lower_quantile) &
                (df_combined[prob_col] <= upper_quantile)
            ]

            df_combined_gray_zone[prob_col] = df_combined_gray_zone[prob_col].clip(0, 1)

            group1_data = df_combined_gray_zone[df_combined_gray_zone['Subgroup'] == 'Group 1'][prob_col]
            group2_data = df_combined_gray_zone[df_combined_gray_zone['Subgroup'] == 'Group 2'][prob_col]

            if len(group1_data) < 5 or len(group2_data) < 5:
                print(f"Not enough data points for meaningful plot in discrepancy {idx + 1}.")
                continue

            sns.kdeplot(group1_data, label=f'Group 1: {val1}', fill=True, alpha=0.8, color=colors[idx], ax=axes[idx // 2, idx % 2])
            sns.kdeplot(group2_data, label=f'Group 2: {val2}', fill=True, alpha=0.8, color=colors[idx], linestyle='--', ax=axes[idx // 2, idx % 2])

            title_labels = ['Biggest discrepancy (RoBERTa-large)', 'Second biggest discrepancy (RoBERTa-large)', 'Second smallest discrepancy (RoBERTa-large)', 'Smallest discrepancy (RoBERTa-large)']
            axes[idx // 2, idx % 2].set_title(title_labels[idx], fontsize=fontsize)
            axes[idx // 2, idx % 2].set_xlabel('Probability Score', fontsize=fontsize)
            axes[idx // 2, idx % 2].set_xlim(lower_quantile, upper_quantile)
            axes[idx // 2, idx % 2].set_ylabel('Estimated density', fontsize=fontsize)
            axes[idx // 2, idx % 2].legend(fontsize=fontsize - 2, loc='upper left')

        plt.tight_layout()
        plt.show()

    def analyze_and_plot(self, quantile_ranges=[(0.25, 0.75)], fontsize=12):
        self.calculate_discrepancies()
        self.print_discrepancy_details()

        for quantile_range in quantile_ranges:
            print(f"\nVisualizing for quantile range: {quantile_range}")
            self.plot_individual_kde(quantile_range, fontsize=fontsize)

        if self.results:
            biggest_discrepancy = self.results[0]
            print("\nBiggest Discrepancy Summary:")
            print(f"  KS Statistic: {biggest_discrepancy['ks_statistic']:.4f}")
            print(f"  p-value: {biggest_discrepancy['p_value']:.4e}")
            return biggest_discrepancy


# Usage example
if __name__ == "__main__":
    filepath = r"C:\Users\minse\Desktop\Programming\FairThresholdOptimization\datasets\train_features.csv"
    feature_columns = ['personality', "length_label"]
    probability_columns = ['roberta_large_openai_detector_probability']

    analysis = FairThresholdAnalysis(filepath, feature_columns, probability_columns)
    analysis.analyze_and_plot(quantile_ranges=[(0.1, 0.9)], fontsize=16)
