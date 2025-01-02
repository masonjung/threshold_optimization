# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import math


class ThresholdOptimizer:
    def __init__(
        self,
        y_true,
        y_pred_proba,
        groups,
        initial_thresholds,
        learning_rate=10**-2,   # Adjusted learning rate
        max_iterations=10**5,
        acceptable_disparity=0.2,
        min_acc_threshold=0.5,
        min_f1_threshold=0.5,  # F1 is for stable performance
        tolerance=1e-4,
        penalty=10            # Penalty term for F1 score below threshold
    ):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.groups = groups
        self.initial_thresholds = initial_thresholds.copy()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.acceptable_ppr_disparity = acceptable_disparity
        self.acceptable_fpr_disparity = acceptable_disparity
        self.acceptable_tpr_disparity = acceptable_disparity
        self.min_acc_threshold = min_acc_threshold
        self.min_f1_threshold = min_f1_threshold
        self.tolerance = tolerance
        self.penalty = penalty
        self.group_indices = {group: (groups == group) for group in np.unique(groups)}
        self.history = {group: {'accuracy': [], 'f1': [], 'threshold': []} for group in np.unique(groups)}
        self.thresholds = self.initial_thresholds.copy()
        self.initial_learning_rate = learning_rate  # Store initial learning rate

    def optimize(self):
        iteration = 0
        previous_thresholds = self.thresholds.copy()

        while iteration < self.max_iterations:
            confusion_matrix_df = pd.DataFrame()
            acc_dict, f1_dict = {}, {}

            # Calculate metrics for each group
            for group, indices in self.group_indices.items():
                group_y_true = self.y_true[indices]
                group_y_pred_proba = self.y_pred_proba[indices]
                threshold = self.thresholds[group]

                # Calculate current predictions
                group_y_pred = group_y_pred_proba >= threshold

                # Calculate accuracy and F1 score
                acc = accuracy_score(group_y_true, group_y_pred)
                f1 = f1_score(group_y_true, group_y_pred, zero_division=1)
                acc_dict[group] = acc
                f1_dict[group] = f1

                # Record the metrics and update confusion matrix
                self.history[group]['accuracy'].append(acc)
                self.history[group]['f1'].append(f1)
                self.history[group]['threshold'].append(threshold)
                confusion_matrix_df = self.update_confusion_matrix(
                    group, group_y_true, group_y_pred, confusion_matrix_df
                )

            # Adjust thresholds using the new gradient computation
            for group in self.group_indices.keys():
                indices = self.group_indices[group]
                group_y_true = self.y_true[indices]
                group_y_pred_proba = self.y_pred_proba[indices]
                threshold = self.thresholds[group]

                gradient = self.compute_gradient(
                    group_y_true, group_y_pred_proba, threshold
                )

                # Check if gradient is effectively zero
                # if abs(gradient) < 1e-7:
                #     print(f"Iteration {iteration}, Group {group}: Gradient is zero, adjusting delta or learning rate.")
                #     # Optionally increase delta or adjust learning rate here if needed

                # Update threshold
                self.thresholds[group] = threshold - self.learning_rate * gradient
                self.thresholds[group] = np.clip(self.thresholds[group], 0.1, 0.9) # range

                # Monitor gradient and threshold updates
                # print(f"Iteration {iteration}, Group {group}, Gradient: {gradient:.7f}, Threshold: {self.thresholds[group]:.7f}")

            # Check convergence
            max_threshold_change = max(
                abs(self.thresholds[group] - previous_thresholds[group]) for group in self.thresholds
            )
            if max_threshold_change < self.tolerance:
                if self.check_fairness(confusion_matrix_df) and self.check_performance_criteria(acc_dict, f1_dict):
                    print(f"Converged after {iteration} iterations.")
                    break

            previous_thresholds = self.thresholds.copy()
            iteration += 1

        return self.thresholds, self.history

    def update_confusion_matrix(self, group, y_true, y_pred, confusion_matrix_df):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        confusion_matrix_df.loc[group, 'True Positives'] = tp
        confusion_matrix_df.loc[group, 'True Negatives'] = tn
        confusion_matrix_df.loc[group, 'False Positives'] = fp
        confusion_matrix_df.loc[group, 'False Negatives'] = fn

        confusion_matrix_df.loc[group, 'PPR'] = (tp + fp) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        confusion_matrix_df.loc[group, 'FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        confusion_matrix_df.loc[group, 'TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        return confusion_matrix_df

    # check fairness 
    def check_fairness(self, confusion_matrix_df):        
        
        # Define acceptable disparity thresholds
        ppr_values = confusion_matrix_df['PPR'].fillna(0).values
        fpr_values = confusion_matrix_df['FPR'].fillna(0).values
        tpr_values = confusion_matrix_df['TPR'].fillna(0).values

        ppr_disparity = ppr_values.max() - ppr_values.min()
        fpr_disparity = fpr_values.max() - fpr_values.min()
        tpr_disparity = tpr_values.max() - tpr_values.min()

        # DP
        if ppr_disparity <= self.acceptable_ppr_disparity and (fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity):
            return True
        else:
            return False
        
    def check_performance_criteria(self, acc_dict, f1_dict):
        # Ensure performance criteria is met for both accuracy and F1 score across groups
        if all(acc >= self.min_acc_threshold for acc in acc_dict.values()) and \
           all(f1 >= self.min_f1_threshold for f1 in f1_dict.values()):
            return True
        else:
            return False

    def compute_gradient(self, group_y_true, group_y_pred_proba, threshold):
        # Calculate current predictions
        group_y_pred = group_y_pred_proba >= threshold

        # Calculate accuracy and F1 score
        acc = accuracy_score(group_y_true, group_y_pred)
        f1 = f1_score(group_y_true, group_y_pred, zero_division=1)

        # Compute loss: negative accuracy (to maximize accuracy)
        loss = -acc

        # Add penalty if F1 score is below the minimum threshold
        if f1 < self.min_f1_threshold:
            loss += self.penalty * (self.min_f1_threshold - f1)

        # Adjust delta
        delta = 0.01  # Increased delta

        threshold_plus = min(threshold + delta, 1)
        threshold_minus = max(threshold - delta, 0)

        # Loss at threshold_plus
        group_y_pred_plus = group_y_pred_proba >= threshold_plus
        acc_plus = accuracy_score(group_y_true, group_y_pred_plus)
        f1_plus = f1_score(group_y_true, group_y_pred_plus, zero_division=1)
        loss_plus = -acc_plus
        if f1_plus < self.min_f1_threshold:
            loss_plus += self.penalty * (self.min_f1_threshold - f1_plus)

        # Loss at threshold_minus
        group_y_pred_minus = group_y_pred_proba >= threshold_minus
        acc_minus = accuracy_score(group_y_true, group_y_pred_minus)
        f1_minus = f1_score(group_y_true, group_y_pred_minus, zero_division=1)
        loss_minus = -acc_minus
        if f1_minus < self.min_f1_threshold:
            loss_minus += self.penalty * (self.min_f1_threshold - f1_minus)

        # Compute numerical gradient
        gradient = (loss_plus - loss_minus) / (2 * delta)

        # Diagnostic prints to check if predictions are changing
        preds = group_y_pred
        preds_plus = group_y_pred_plus
        preds_minus = group_y_pred_minus
        changes_plus = np.sum(preds != preds_plus)
        changes_minus = np.sum(preds != preds_minus)
        # print(f"Group Threshold: {threshold:.2f}, Delta: {delta}, Changes +delta: {changes_plus}, Changes -delta: {changes_minus}")

        return gradient



######################## RUN

# import df
dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv")

#split by train and tesxt <- change this later
df = dataset.sample(frac=1, random_state=42)



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
    f"{length}_{personality}"
    for length, personality in zip(length_groups, personality_groups)
]).values

# groups = pd.Series([
#     f"{length}_{formality}_{sentiment}_{personality}"
#     for length, formality, sentiment, personality in zip(length_groups, formality_groups, sentiment_groups, personality_groups)
# ]).values

# Prepare true labels and predicted probabilities
y_true = df['AI_written']  # True labels
y_pred_proba = df['roberta_large_openai_detector_probability'].values     # Predicted probabilities the probability is learned from one model

# Initial thresholds (set to 0.5 for all groups)
initial_thresholds = {group: 0.5 for group in np.unique(groups)}

# Create an instance of ThresholdOptimizer
optimizer = ThresholdOptimizer(
    y_true,
    y_pred_proba,
    groups,
    initial_thresholds,
    learning_rate=10**-2,
    max_iterations=10**4,
    acceptable_disparity=0.1,  # Adjust based on your fairness criteria
    min_acc_threshold=0.5,         # Set realistic minimum accuracy
    min_f1_threshold=0.5,           # Set realistic minimum F1 score
    tolerance=1e-4,  # Decrease tolerance for stricter convergence criteria
    penalty=20  # Increase penalty to enforce stricter updates
)

# Optimize thresholds using gradient-based method
thresholds, history = optimizer.optimize()


# No need to convert thresholds to a list if you intend to use it as a dictionary
# Remove the optimized_thresholds_list entirely or use it correctly if needed

# Print the list of optimized thresholds (optional)
print("\nOptimized Thresholds:")
for group, threshold in thresholds.items():
    print(f"Group: {group}, Threshold: {threshold:.7f}")

##################################TEST

# Load test dataset
test_dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t4_features.csv")

# Split test dataset by 'source'
unique_sources = test_dataset['source'].unique()

def calculate_discrepancies(test_y_true, test_y_pred, features):
    feature_discrepancies = {}

    for feature_name, feature_values in features.items():
        unique_feature_values = np.unique(feature_values)
        tpr_values, tnr_values, fpr_values, fnr_values = [], [], [], []

        for feature_value in unique_feature_values:
            group_indices = (feature_values == feature_value)
            TP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 1))
            TN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 0))
            FP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 0))
            FN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 1))

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

            tpr_values.append(TPR)
            tnr_values.append(TNR)
            fpr_values.append(FPR)
            fnr_values.append(FNR)

        tpr_discrepancy = max(tpr_values) - min(tpr_values) if tpr_values else 0
        tnr_discrepancy = max(tnr_values) - min(tnr_values) if tnr_values else 0
        fpr_discrepancy = max(fpr_values) - min(fpr_values) if fpr_values else 0
        fnr_discrepancy = max(fnr_values) - min(fnr_values) if fnr_values else 0

        feature_discrepancies[feature_name] = {
            "TPR": tpr_discrepancy,
            "TNR": tnr_discrepancy,
            "FPR": fpr_discrepancy,
            "FNR": fnr_discrepancy,
            "BER": (fpr_discrepancy + fnr_discrepancy) / 2
        }

    return feature_discrepancies

detector_probabilities = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability', 'GPT4o-mini_probability']

# Main loop with added discrepancy calculations
for source in unique_sources:
    source_dataset = test_dataset[test_dataset['source'] == source]

    for detector in detector_probabilities:
        if detector not in source_dataset.columns:
            continue

        test_y_true = source_dataset['AI_written']
        test_y_pred_proba = source_dataset[detector].values

        # Ensure group definitions match training
        test_groups = pd.Series([
            f"{length}_{personality}"
            for length, personality in zip(
                pd.cut(source_dataset['text_length'], bins=[0, 1000, 2500, np.inf], labels=['short', 'medium', 'long']).astype(str).values,
                source_dataset['personality'].fillna('unknown').astype(str).values
            )
        ]).values

        test_y_pred = np.zeros_like(test_y_true)
        for group in np.unique(test_groups):
            group_indices = (test_groups == group)
            threshold = thresholds.get(group, 0.2)  # Use thresholds dictionary
            test_y_pred[group_indices] = test_y_pred_proba[group_indices] >= threshold

        test_accuracy = accuracy_score(test_y_true, test_y_pred)
        test_fpr = np.sum((test_y_pred == 1) & (test_y_true == 0)) / np.sum(test_y_true == 0) if np.sum(test_y_true == 0) > 0 else 0
        test_fnr = np.sum((test_y_pred == 0) & (test_y_true == 1)) / np.sum(test_y_true == 1) if np.sum(test_y_true == 1) > 0 else 0
        test_ber = (test_fpr + test_fnr) / 2

        features = {
            'length': pd.cut(source_dataset['text_length'], bins=[0, 1000, 2500, np.inf], labels=['short', 'medium', 'long']).astype(str).values,
            'personality': source_dataset['personality'].astype(str).values
        }

        feature_discrepancies = calculate_discrepancies(test_y_true, test_y_pred, features)

        with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results_updated_0.1.txt", "a") as f:
            f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"False Positive Rate (FPR): {test_fpr:.4f}\n")
            f.write(f"Balanced Error Rate (BER): {test_ber:.4f}\n")
            for group in np.unique(test_groups):
                threshold = thresholds.get(group, 0.5)  # Use thresholds dictionary
                f.write(f"  Group: {group}, Threshold: {threshold:.4f}\n")
            f.write(f"Discrepancies by Feature:\n")
            for feature_name, discrepancies in feature_discrepancies.items():
                f.write(f"  Feature: {feature_name.capitalize()}\n")
                f.write(f"   BER Discrepancy: {discrepancies['BER']:.4f}\n")








# # Move the results to the list
# optimized_thresholds_list = []
# for group, threshold in thresholds.items():
#     optimized_thresholds_list.append({'group': group, 'threshold': threshold})

# # Print the list of optimized thresholds
# print("\nOptimized Thresholds:")
# for group, threshold in thresholds.items():
#     print(f"Group: {group}, Threshold: {threshold:.7f}")



# ##################################TEST

# # need to apply the generated thresold to the test dataset
# # Load test dataset
# test_dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t4_features.csv")

# # Split test dataset by 'source'
# unique_sources = test_dataset['source'].unique()

# # Calculate and return discrepancies for each feature
# def calculate_discrepancies(test_y_true, test_y_pred, features):
#     feature_discrepancies = {}

#     for feature_name, feature_values in features.items():
#         unique_feature_values = np.unique(feature_values)
#         tpr_values, tnr_values, fpr_values, fnr_values = [], [], [], []

#         for feature_value in unique_feature_values:
#             group_indices = (feature_values == feature_value)
#             TP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 1))
#             TN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 0))
#             FP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 0))
#             FN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 1))

#             TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
#             TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
#             FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
#             FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

#             tpr_values.append(TPR)
#             tnr_values.append(TNR)
#             fpr_values.append(FPR)
#             fnr_values.append(FNR)

#         tpr_discrepancy = max(tpr_values) - min(tpr_values) if tpr_values else 0
#         tnr_discrepancy = max(tnr_values) - min(tnr_values) if tnr_values else 0
#         fpr_discrepancy = max(fpr_values) - min(fpr_values) if fpr_values else 0
#         fnr_discrepancy = max(fnr_values) - min(fnr_values) if fnr_values else 0

#         feature_discrepancies[feature_name] = {
#             "TPR": tpr_discrepancy,
#             "TNR": tnr_discrepancy,
#             "FPR": fpr_discrepancy,
#             "FNR": fnr_discrepancy,
#             "BER": (fpr_discrepancy + fnr_discrepancy) / 2
#         }

#     return feature_discrepancies

# detector_probabilities = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability', 'GPT4o-mini_probability']


# # Main loop with added discrepancy calculations
# for source in unique_sources:
#     source_dataset = test_dataset[test_dataset['source'] == source]

#     for detector in detector_probabilities:
#         if detector not in source_dataset.columns:
#             continue

#         test_y_true = source_dataset['AI_written']
#         test_y_pred_proba = source_dataset[detector].values

#         test_groups = pd.Series([
#             f"{length}_{sentiment}"
#             for length, sentiment in zip(
#                 pd.cut(source_dataset['text_length'], bins=[0, 1000, 2500, np.inf], labels=['short', 'medium', 'long']).astype(str).values,
#                 source_dataset['sentiment_label'].fillna('neutral').astype(str).values,
#                 # pd.cut(source_dataset['formality'], bins=[0, 50, np.inf], labels=['informal', 'formal']).astype(str).values
#             )
#         ]).values

#         test_y_pred = np.zeros_like(test_y_true)
#         for group in np.unique(test_groups):
#             group_indices = (test_groups == group)
#             threshold = optimized_thresholds_list.get(group, 0.5)
#             test_y_pred[group_indices] = test_y_pred_proba[group_indices] >= threshold

#         test_accuracy = accuracy_score(test_y_true, test_y_pred)
#         test_fpr = np.sum((test_y_pred == 1) & (test_y_true == 0)) / np.sum(test_y_true == 0) if np.sum(test_y_true == 0) > 0 else 0
#         test_fnr = np.sum((test_y_pred == 0) & (test_y_true == 1)) / np.sum(test_y_true == 1) if np.sum(test_y_true == 1) > 0 else 0
#         test_ber = (test_fpr + test_fnr) / 2

#         features = {
#             'length': pd.cut(source_dataset['text_length'], bins=[0, 1000, 2500, np.inf], labels=['short', 'medium', 'long']).astype(str).values,
#             # 'sentiment': source_dataset['sentiment_label'].astype(str).values,
#             # 'formality': pd.cut(source_dataset['formality'], bins=[0, 50, np.inf], labels=['informal', 'formal']).astype(str).values,
#             'personality': source_dataset['personality'].astype(str).values
#         }

#         feature_discrepancies = calculate_discrepancies(test_y_true, test_y_pred, features)

#         with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results_updated2.txt", "a") as f:
#             f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
#             f.write(f"Accuracy: {test_accuracy:.4f}\n")
#             f.write(f"False Positive Rate (FPR): {test_fpr:.4f}\n")
#             f.write(f"Balanced Error Rate (BER): {test_ber:.4f}\n")
#             for group in np.unique(test_groups):
#                 threshold = thresholds.get(group, 0.5)  # Default to 0.5 if group not found
#                 f.write(f"  Group: {group}, Threshold: {threshold:.4f}\n")
#             f.write(f"Discrepancies by Feature:\n")
#             for feature_name, discrepancies in feature_discrepancies.items():
#                 f.write(f"  Feature: {feature_name.capitalize()}\n")
#                 f.write(f"   BER Discrepancy: {discrepancies['BER']:.4f}\n")



