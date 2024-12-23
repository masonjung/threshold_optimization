# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score


import math

# Adjust the accuracy and FPR to round down to four decimal places
def truncate_to_decimal(value, decimals):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

# Interpreting equaltion
def equation_fairness(Prob_a,Prob_b):
    rule = 0.8 # 0.8 for relaxed fairness; 1.0 is strict fairness
    greater = max(Prob_a, Prob_b)
    smaller = min(Prob_a, Prob_b)
    evaluation = (smaller/greater) > rule # if the ratio exceeds, it is considered fair = True
    return evaluation
    # Based on Feldman, M., etc. (2015) Certifying and removing disparate impact.

# Equalized Odds
def Equalized_Odds(confusion_matrix_df, group_a_list, group_b_list):
    # Calculate combined FPR and TPR for groups in group_a_list
    FP_a = confusion_matrix_df.loc[group_a_list, 'False Positives'].sum()
    TN_a = confusion_matrix_df.loc[group_a_list, 'True Negatives'].sum()
    TP_a = confusion_matrix_df.loc[group_a_list, 'True Positives'].sum()
    FN_a = confusion_matrix_df.loc[group_a_list, 'False Negatives'].sum()

    FPR_a = FP_a / (FP_a + TN_a) if (FP_a + TN_a) > 0 else 0
    TPR_a = TP_a / (TP_a + FN_a) if (TP_a + FN_a) > 0 else 0

    # Calculate combined FPR and TPR for groups in group_b_list
    FP_b = confusion_matrix_df.loc[group_b_list, 'False Positives'].sum()
    TN_b = confusion_matrix_df.loc[group_b_list, 'True Negatives'].sum()
    TP_b = confusion_matrix_df.loc[group_b_list, 'True Positives'].sum()
    FN_b = confusion_matrix_df.loc[group_b_list, 'False Negatives'].sum()

    FPR_b = FP_b / (FP_b + TN_b) if (FP_b + TN_b) > 0 else 0
    TPR_b = TP_b / (TP_b + FN_b) if (TP_b + FN_b) > 0 else 0

    # Evaluate fairness for both FPR and TPR
    fpr_fair = equation_fairness(FPR_a, FPR_b)
    tpr_fair = equation_fairness(TPR_a, TPR_b)

    # Print results for fairness checks
    print(f"FPR Fairness between groups {group_a_list} and {group_b_list}: {'Fair' if fpr_fair else 'Unfair'}")
    print(f"TPR Fairness between groups {group_a_list} and {group_b_list}: {'Fair' if tpr_fair else 'Unfair'}")

    # Combine FPR and TPR fairness
    fair = fpr_fair and tpr_fair

    # Return combined fairness result
    print(f"Overall Equalized Odds fairness between groups {group_a_list} and {group_b_list}: {'Fair' if fair else 'Unfair'}")
    return fair

# Demographic Parity
def Demographic_Parity(confusion_matrix_df):
    # Assuming confusion_matrix_df has a 'Group' index and a column 'PPR' for Positive Prediction Rate

    # Iterate through pairs of groups to compare their PPRs
    groups = confusion_matrix_df.index
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_a, group_b = groups[i], groups[j]
            ppr_a = confusion_matrix_df.loc[group_a, 'PPR']
            ppr_b = confusion_matrix_df.loc[group_b, 'PPR']

            # Evaluate fairness between each pair of groups
            fair = equation_fairness(ppr_a, ppr_b)
            print(f"Fairness between Group {group_a} and Group {group_b}: {'Fair' if fair else 'Unfair'}")
    return fair

def calculate_optimal_threshold(file_path, true_label_column, pred_columns, fpr_limit=0.1):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Extract true labels and predicted probabilities
    true_labels = df[true_label_column]
    avg_pred = df[pred_columns].mean(axis=1)

    # Compute the ROC curve and AUROC
    fpr, tpr, thresholds = roc_curve(true_labels, avg_pred)
    auroc = roc_auc_score(true_labels, avg_pred)

    # Filter thresholds for FPR < fpr_limit
    valid_indices = np.where(fpr < fpr_limit)[0]

    # Identify the optimal threshold: highest TPR under FPR constraint
    optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]
    optimal_threshold = thresholds[optimal_idx]

    # Compile results
    results = {
        "Optimal AUROC Threshold": optimal_threshold,
        "AUROC": auroc,
        "FPR at Optimal Threshold": fpr[optimal_idx],
        "TPR at Optimal Threshold": tpr[optimal_idx]
    }

    return results




# def calculate_metrics_by_group(data, group_col, thresholds, classifiers):
#     """
#     Calculates metrics by a specific group column (e.g., formality_group).
#     Returns: results[group_value][classifier][threshold_key] = {TP, TN, FP, FN, ACC, FPR}
#     """
#     results = {}
#     unique_groups = data[group_col].unique()
#     for grp in unique_groups:
#         grp_data = data[data[group_col] == grp].copy()
        
#         if grp_data.empty:
#             continue
        
#         for classifier in classifiers:
#             if classifier not in grp_data.columns:
#                 continue
            
#             if grp not in results:
#                 results[grp] = {}
#             if classifier not in results[grp]:
#                 results[grp][classifier] = {}
            
#             for i, threshold in enumerate(thresholds, 1):
#                 pred_col = f'pred_{i}'
#                 actual_col = f'actual_{i}'
                
#                 grp_data[pred_col] = grp_data[classifier].apply(lambda x: 1 if x >= threshold else 0)
#                 grp_data[actual_col] = grp_data['AI_written'].apply(lambda x: 1 if x == 1 else 0)
                
#                 TP = len(grp_data[(grp_data[pred_col] == 1) & (grp_data[actual_col] == 1)])
#                 TN = len(grp_data[(grp_data[pred_col] == 0) & (grp_data[actual_col] == 0)])
#                 FP = len(grp_data[(grp_data[pred_col] == 1) & (grp_data[actual_col] == 0)])
#                 FN = len(grp_data[(grp_data[pred_col] == 0) & (grp_data[actual_col] == 1)])
                
#                 denom = TP + TN + FP + FN
#                 ACC = (TP + TN) / denom if denom > 0 else 0
#                 FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                
#                 results[grp][classifier][f'threshold_{i}'] = {
#                     'TP': TP,
#                     'TN': TN,
#                     'FP': FP,
#                     'FN': FN,
#                     'ACC': ACC,
#                     'FPR': FPR
#                 }
#     return results

# def calculate_statistical_discrepancy(metrics_dict, classifiers, thresholds):
#     """
#     Calculates the biggest statistical parity discrepancy for each feature group.
#     """
#     discrepancies = {}
#     groups = list(metrics_dict.keys())
#     if len(groups) < 2:
#         return discrepancies  # Not enough subgroups to measure discrepancy
    
#     for classifier in classifiers:
#         for i, threshold in enumerate(thresholds, 1):
#             threshold_key = f'threshold_{i}'
#             positive_rates = []
            
#             for grp in groups:
#                 clf_data = metrics_dict.get(grp, {}).get(classifier, {})
#                 if threshold_key in clf_data:
#                     TP = clf_data[threshold_key]['TP']
#                     FP = clf_data[threshold_key]['FP']
#                     FN = clf_data[threshold_key]['FN']
#                     TN = clf_data[threshold_key]['TN']
                    
#                     total = TP + FP + FN + TN
#                     positive_rate = (TP + FP) / total if total > 0 else 0
#                     positive_rates.append(positive_rate)
            
#             if len(positive_rates) < 2:
#                 continue
            
#             statistical_discrepancy = max(positive_rates) - min(positive_rates)
            
#             if classifier not in discrepancies:
#                 discrepancies[classifier] = {}
#             discrepancies[classifier][threshold_key] = {
#                 'Statistical Discrepancy': statistical_discrepancy
#             }
#     return discrepancies

def calculate_metrics_by_group(data, group_col, thresholds, classifiers):
    """
    Calculates metrics by a specific group column (e.g., formality_group).
    Returns: results[group_value][classifier][threshold_key] = {TP, TN, FP, FN, etc.}
    """
    results = {}
    unique_groups = data[group_col].unique()
    for grp in unique_groups:
        grp_data = data[data[group_col] == grp].copy()
        
        if grp_data.empty:
            continue
        
        for classifier in classifiers:
            if classifier not in grp_data.columns:
                continue
            
            if grp not in results:
                results[grp] = {}
            if classifier not in results[grp]:
                results[grp][classifier] = {}
            
            for i, threshold in enumerate(thresholds, 1):
                pred_col = f'pred_{i}'
                actual_col = f'actual_{i}'
                
                grp_data[pred_col] = grp_data[classifier].apply(lambda x: 1 if x >= threshold else 0)
                grp_data[actual_col] = grp_data['AI_written'].apply(lambda x: 1 if x == 1 else 0)
                
                TP = len(grp_data[(grp_data[pred_col] == 1) & (grp_data[actual_col] == 1)])
                TN = len(grp_data[(grp_data[pred_col] == 0) & (grp_data[actual_col] == 0)])
                FP = len(grp_data[(grp_data[pred_col] == 1) & (grp_data[actual_col] == 0)])
                FN = len(grp_data[(grp_data[pred_col] == 0) & (grp_data[actual_col] == 1)])
                
                denom = TP + TN + FP + FN
                ACC = (TP + TN) / denom if denom > 0 else 0
                FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
                PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
                NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
                
                results[grp][classifier][f'threshold_{i}'] = {
                    'TP': TP,
                    'TN': TN,
                    'FP': FP,
                    'FN': FN,
                    'ACC': ACC,
                    'FPR': FPR,
                    'FNR': FNR,
                    'PPV': PPV,
                    'NPV': NPV
                }
    return results

def calculate_fairness_discrepancy(metrics_dict, classifiers, thresholds):
    """
    Calculates fairness metrics (TPR, FPR, TNR, FNR) for the given classifiers and thresholds.
    Returns the metric with the greatest disparity for each classifier and threshold.
    """
    fairness_metrics = ['TPR', 'FPR', 'TNR', 'FNR']
    discrepancies = {}
    groups = list(metrics_dict.keys())
    if len(groups) < 2:
        return discrepancies  # Not enough subgroups to measure discrepancy
    
    for classifier in classifiers:
        for i, threshold in enumerate(thresholds, 1):
            threshold_key = f'threshold_{i}'
            metric_discrepancies = {metric: [] for metric in fairness_metrics}
            
            for grp in groups:
                clf_data = metrics_dict.get(grp, {}).get(classifier, {})
                if threshold_key in clf_data:
                    TP = clf_data[threshold_key]['TP']
                    FP = clf_data[threshold_key]['FP']
                    FN = clf_data[threshold_key]['FN']
                    TN = clf_data[threshold_key]['TN']
                    
                    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
                    FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
                    
                    metric_discrepancies['TPR'].append(TPR)
                    metric_discrepancies['FPR'].append(FPR)
                    metric_discrepancies['TNR'].append(TNR)
                    metric_discrepancies['FNR'].append(FNR)
            
            # Calculate discrepancies for each metric
            metric_disparities = {metric: (max(values) - min(values)) if values else 0 
                                  for metric, values in metric_discrepancies.items()}
            
            # Find the greatest discrepancy
            max_discrepancy_metric = max(metric_disparities, key=metric_disparities.get)
            max_discrepancy_value = metric_disparities[max_discrepancy_metric]
            
            if classifier not in discrepancies:
                discrepancies[classifier] = {}
            discrepancies[classifier][threshold_key] = {
                'Greatest Discrepancy': max_discrepancy_value,
                'Metric': max_discrepancy_metric
            }
    return discrepancies

# second method
def calculate_statistical_discrepancy(metrics_dict, classifiers, thresholds):
    discrepancies = {}
    groups = list(metrics_dict.keys())
    if len(groups) < 2:
        return discrepancies  # Not enough subgroups to measure discrepancy
    
    for classifier in classifiers:
        for i, threshold in enumerate(thresholds, 1):
            threshold_key = f'threshold_{i}'
            TPRs = []
            FPRs = []
            
            for grp in groups:
                clf_data = metrics_dict.get(grp, {}).get(classifier, {})
                if threshold_key in clf_data:
                    TP = clf_data[threshold_key]['TP']
                    FP = clf_data[threshold_key]['FP']
                    FN = clf_data[threshold_key]['FN']
                    TN = clf_data[threshold_key]['TN']
                    
                    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                    TPRs.append(TPR)
                    FPRs.append(FPR)
            
            if len(TPRs) < 2 or len(FPRs) < 2:
                continue
            
            TPR_discrepancy = max(TPRs) - min(TPRs)
            FPR_discrepancy = max(FPRs) - min(FPRs)
            greatest_discrepancy = max(TPR_discrepancy, FPR_discrepancy)
            
            if classifier not in discrepancies:
                discrepancies[classifier] = {}
            discrepancies[classifier][threshold_key] = {
                'Statistical Discrepancy': greatest_discrepancy
            }
    return discrepancies

# Run the optimization method 2
file_path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv"
true_label_column = 'AI_written'
pred_columns = [
    # 'roberta_base_openai_detector_probability',
    'roberta_large_openai_detector_probability',
    # 'radar_probability'
]

results = calculate_optimal_threshold(file_path, true_label_column, pred_columns)
print(results)

AUROC_threshold = results["Optimal AUROC Threshold"]
print(AUROC_threshold)


######## Threshold Optimizer
class ThresholdOptimizer:
    def __init__(
        self,
        y_true,
        y_pred_proba,
        groups,
        initial_thresholds,
        learning_rate=10**-2,   # Adjusted learning rate
        max_iterations=10**5,

        relaxation_disparity=0.2, # relaxation standard

        min_acc_threshold=0.5, # do we need this?
        min_f1_threshold=0.5,  # do we need this?
        tolerance=1e-4,
        penalty=20            # Penalty term for F1 score below threshold
    ):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.groups = groups
        self.initial_thresholds = initial_thresholds.copy()
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations

        self.relaxation_disparity = relaxation_disparity

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
                if abs(gradient) < 1e-7:
                    print(f"Iteration {iteration}, Group {group}: Gradient is zero, adjusting delta or learning rate.")
                    # Optionally increase delta or adjust learning rate here if needed

                # Update threshold
                self.thresholds[group] = threshold - self.learning_rate * gradient
                self.thresholds[group] = np.clip(self.thresholds[group], 0.00005, 0.99995) # 7 decimal points

                # Monitor gradient and threshold updates
                print(f"Iteration {iteration}, Group {group}, Gradient: {gradient:.7f}, Threshold: {self.thresholds[group]:.7f}")

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
        
        total_instances = len(y_true)
        ppr = (tp + fp) / total_instances if total_instances > 0 else 0

        confusion_matrix_df.loc[group, 'True Positives'] = tp
        confusion_matrix_df.loc[group, 'True Negatives'] = tn
        confusion_matrix_df.loc[group, 'False Positives'] = fp
        confusion_matrix_df.loc[group, 'False Negatives'] = fn

        confusion_matrix_df.loc[group, 'FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        confusion_matrix_df.loc[group, 'TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        confusion_matrix_df.loc[group, 'PPR'] = ppr
        return confusion_matrix_df
    
    # Fairness metrics
    def check_fairness(self, confusion_matrix_df):
        relaxation = self.relaxation_disparity
        fpr_values = confusion_matrix_df['FPR'].fillna(0).values
        tpr_values = confusion_matrix_df['TPR'].fillna(0).values
        ppr_values = confusion_matrix_df['PPR'].fillna(0).values

        fpr_disparity = fpr_values.max() - fpr_values.min()
        tpr_disparity = tpr_values.max() - tpr_values.min()
        ppr_disparity = ppr_values.max() - ppr_values.min()
        
        dp_condition = ppr_disparity <= relaxation # demographic parity or statistical parity
        eo_condition = (fpr_disparity <= relaxation) and (tpr_disparity <= relaxation) # euqalized odds
        
        if dp_condition and eo_condition:
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
        print(f"Group Threshold: {threshold:.8f}, Delta: {delta}, Changes +delta: {changes_plus}, Changes -delta: {changes_minus}")

        return gradient

    def grid_search_thresholds(self):
        possible_thresholds = np.linspace(0, 1, num=100)
        best_thresholds = {}
        for group in self.group_indices.keys():
            best_acc = -np.inf
            best_threshold = 0.5
            group_indices = self.group_indices[group]
            group_y_true = self.y_true[group_indices]
            group_y_pred_proba = self.y_pred_proba[group_indices]
            for threshold in possible_thresholds:
                group_y_pred = group_y_pred_proba >= threshold
                acc = accuracy_score(group_y_true, group_y_pred)
                f1 = f1_score(group_y_true, group_y_pred, zero_division=1)
                if f1 >= self.min_f1_threshold and acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold
            best_thresholds[group] = best_threshold
            print(f"Group: {group}, Best Threshold: {best_threshold:.4f}, Best Accuracy: {best_acc:.4f}")
        return best_thresholds



######################## RUN

# df
dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv")
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



# Combine groups into a single group label 1
groups = pd.Series([
    f"{length}_{formality}_{sentiment}_{personality}"
    for length, formality, sentiment, personality in zip(length_groups, formality_groups, sentiment_groups, personality_groups)
]).values

# Prepare true labels and predicted probabilities
y_true = df['AI_written'].apply(lambda x: 1 if x == 'AI' else 0).values  # True labels
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
    max_iterations=5000,
    relaxation_disparity=0.2,  # relaxation <- 0.2 or 0 <- this can be used for the figure drawing
    min_acc_threshold=0.5,         # accuracy
    min_f1_threshold=0.5,           # f1
    tolerance=1e-5,  # Decrease tolerance for stricter convergence criteria
    penalty=20  # Increase penalty to enforce stricter updates
)

# Optimize thresholds using gradient-based method
thresholds, history = optimizer.optimize()

# If thresholds still do not change, use grid search as an alternative
if all(threshold == 0.5 for threshold in thresholds.values()):
    print("Thresholds did not change using gradient descent. Switching to grid search.")
    thresholds = optimizer.grid_search_thresholds()

# Move the results to the list
optimized_thresholds_list = [] ##### This is the list of thresholds that will be used in the test dataset
for group, threshold in thresholds.items():
    optimized_thresholds_list.append({'group': group, 'threshold': threshold})

# Print the list of optimized thresholds
print("\nOptimized Thresholds:")
for group, threshold in thresholds.items():
    print(f"Group: {group}, Threshold: {threshold:.7f}")

# check if the thresholds were adjusted
print(optimized_thresholds_list)

##################################TEST

# need to apply the generated thresold to the test dataset
# Load test dataset
test_dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t4_features.csv")

# Split test dataset by 'source'
unique_sources = test_dataset['source'].unique()

for source in unique_sources:
    source_dataset = test_dataset[test_dataset['source'] == source]

    # Split by different detector probabilities
    detector_probabilities = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability', "GPT4o-mini_probability"]

    for detector in detector_probabilities:
        if detector not in source_dataset.columns:
            continue

        # Prepare test dataset groups
        test_length_groups = pd.cut(
            source_dataset['text_length'],
            bins=[0, 1000, 2500, np.inf],
            labels=['short', 'medium', 'long']
        ).astype(str).values

        test_sentiment_groups = source_dataset['sentiment_label'].astype(str).values

        test_formality_groups = pd.cut(
            source_dataset['formality'],
            bins=[0, 50, np.inf],
            labels=['informal', 'formal']
        ).astype(str).values

        test_personality_groups = source_dataset['personality'].astype(str).values

        feature_groups = [test_length_groups, test_formality_groups, test_sentiment_groups, test_personality_groups]

        # Combine groups into a single group label for test dataset
        test_groups = pd.Series([
            f"{length}_{formality}_{sentiment}_{personality}"
            for length, formality, sentiment, personality in zip(test_length_groups, test_formality_groups, test_sentiment_groups, test_personality_groups)
        ]).values

        # Prepare true labels and predicted probabilities for test dataset
        test_y_true = source_dataset['AI_written'].apply(lambda x: 1 if x == 'AI' else 0).values
        test_y_pred_proba = source_dataset[detector].values

        # Apply optimized thresholds to test dataset
        test_y_pred = np.zeros_like(test_y_true)
        for group in np.unique(test_groups):
            group_indices = (test_groups == group)
            threshold = thresholds.get(group, 0.5)  # Default to 0.5 if group not found
            test_y_pred[group_indices] = test_y_pred_proba[group_indices] >= threshold

        # Calculate and print performance metrics for the current source and detector
        test_accuracy = accuracy_score(test_y_true, test_y_pred)
        test_fpr = np.sum((test_y_pred == 1) & (test_y_true == 0)) / np.sum(test_y_true == 0) if np.sum(test_y_true == 0) > 0 else 0
        test_fnr = np.sum((test_y_pred == 0) & (test_y_true == 1)) / np.sum(test_y_true == 1) if np.sum(test_y_true == 1) > 0 else 0
        test_ber = (test_fpr + test_fnr) / 2
 
        # Initialize discrepancy tracking for EACH FEATURE
        feature_discrepancies = {}

        # Iterate over individual features to calculate discrepancies
        features = {
            'length': test_length_groups,
            'sentiment': test_sentiment_groups,
            'formality': test_formality_groups,
            'personality': test_personality_groups
        }

        for feature_name, feature_values in features.items():
            unique_feature_values = np.unique(feature_values)
            tpr_values, tnr_values, fpr_values, fnr_values = [], [], [], []

            # Calculate rates for each feature group
            for feature_value in unique_feature_values:
                group_indices = (feature_values == feature_value)
                TP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 1))
                TN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 0))
                FP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 0))
                FN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 1))

                # Calculate TPR, TNR, FPR, and FNR
                TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate
                TNR = TN / (TN + FP) if (TN + FP) > 0 else 0  # True Negative Rate
                FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
                FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
                
                tpr_values.append(TPR)
                tnr_values.append(TNR)
                fpr_values.append(FPR)
                fnr_values.append(FNR)

            # Calculate discrepancies for each rate
            tpr_discrepancy = max(tpr_values) - min(tpr_values) if tpr_values else 0
            tnr_discrepancy = max(tnr_values) - min(tnr_values) if tnr_values else 0
            fpr_discrepancy = max(fpr_values) - min(fpr_values) if fpr_values else 0
            fnr_discrepancy = max(fnr_values) - min(fnr_values) if fnr_values else 0

            # Store discrepancies
            feature_discrepancies[feature_name] = {
                "TPR": tpr_discrepancy,
                "TNR": tnr_discrepancy,
                "FPR": fpr_discrepancy,
                "FNR": fnr_discrepancy,
                "BER": (fpr_discrepancy + fnr_discrepancy) / 2
            }

        
        # Write discrepancies to the results file
        with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results_updated2.txt", "a") as f:
            f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
            # Apply truncation to ACC and FPR
            f.write(f"Accuracy: {truncate_to_decimal(test_accuracy, 4):.4f}\n")
            f.write(f"False Positive Rate (FPR): {truncate_to_decimal(test_fpr, 4):.4f}\n")
            f.write(f"Balanced Error Rate (NER): {truncate_to_decimal(test_fpr, 4):.4f}\n")
            f.write(f"Discrepancies by Feature:\n")
            for feature_name, discrepancies in feature_discrepancies.items():
                f.write(f"  Feature: {feature_name.capitalize()}\n")
                # Write BER discrepancy with truncation
                f.write(f" BER Discrepancy: {truncate_to_decimal(discrepancies['BER'], 4):.4f}\n")

        # # Write discrepancies to the results file
        # with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results_updated2.txt", "a") as f:
        #     f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
        #     f.write(f"Accuracy: {test_accuracy:.4f}\n")
        #     f.write(f"False Positive Rate (FPR): {test_fpr:.4f}\n")
        #     f.write(f"Discrepancies by Feature:\n")
        #     for feature_name, discrepancies in feature_discrepancies.items():
        #         f.write(f"  Feature: {feature_name.capitalize()}\n")
        #         f.write(f" BER Discrepancy: {discrepancies['BER']:.4f}\n")



        # # Initialize discrepancy tracking
        # feature_discrepancies = {}

        # # Iterate over individual features to calculate discrepancies
        # features = {
        #     'length': test_length_groups,
        #     'sentiment': test_sentiment_groups,
        #     'formality': test_formality_groups,
        #     'personality': test_personality_groups
        # }

        # for feature_name, feature_values in features.items():
        #     unique_feature_values = np.unique(feature_values)
        #     fpr_values, fnr_values, tpr_values, tnr_values = [], [], [], []
            
        #     # Calculate rates for each feature group
        #     for feature_value in unique_feature_values:
        #         group_indices = (feature_values == feature_value)
        #         TP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 1))
        #         TN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 0))
        #         FP = np.sum((test_y_pred[group_indices] == 1) & (test_y_true[group_indices] == 0))
        #         FN = np.sum((test_y_pred[group_indices] == 0) & (test_y_true[group_indices] == 1))
                
        #         FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        #         FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
        #         TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        #         TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
                
        #         fpr_values.append(FPR)
        #         fnr_values.append(FNR)
        #         tpr_values.append(TPR)
        #         tnr_values.append(TNR)
                
    
        #         fpr_discrepancy = max(fpr_values) - min(fpr_values)
        #         fnr_discrepancy = max(fnr_values) - min(fnr_values)
        #         tpr_discrepancy = max(tpr_values) - min(tpr_values)
        #         tnr_discrepancy = max(tnr_values) - min(tnr_values)
                
        #         max_discrepancy = max(fpr_discrepancy, fnr_discrepancy, tpr_discrepancy, tnr_discrepancy)
        #         feature_discrepancies[feature_name] = max_discrepancy

        # # Find the feature with the biggest discrepancy
        # biggest_discrepancy_feature = max(feature_discrepancies, key=feature_discrepancies.get, default=None)
        # biggest_discrepancy_value = feature_discrepancies.get(biggest_discrepancy_feature, 0)

        # # Write discrepancies to the results file
        # with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results_updated2.txt", "a") as f:
        #     f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
        #     f.write(f"Accuracy: {test_accuracy:.4f}\n")
        #     f.write(f"False Positive Rate (FPR): {test_fpr:.4f}\n")
        #     f.write(f"Discrepancies by Feature:\n")
        #     for feature_name, discrepancy in feature_discrepancies.items():
        #         f.write(f"{feature_name.capitalize()} Discrepancy: {discrepancy:.4f}\n")
        #         f.write(f"Biggest Discrepancy: {biggest_discrepancy_feature.capitalize()} ({biggest_discrepancy_value:.4f})\n")
        #         f.write(f"Discrepancy Metric: {max(fpr_discrepancy, fnr_discrepancy, tpr_discrepancy, tnr_discrepancy):.4f}\n")
            
            
            
            
            
            # for feature_name, discrepancy in feature_discrepancies.items():
            #         f.write(f"{feature_name.capitalize()} Discrepancy: {discrepancy:.3f}\n")
            # f.write(f"Biggest Discrepancy: {biggest_discrepancy_feature.capitalize()} ({biggest_discrepancy_value:.3f})\n")

