# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score

# Interpreting equaltion
def equation_fairness(Prob_a,Prob_b):
    rule = 0.8 # 0.8 for relaxed fairness; 1.0 is strict fairness
    greater = max(Prob_a, Prob_b)
    smaller = min(Prob_a, Prob_b)
    evaluation = (smaller/greater) > rule # if the ratio exceeds, it is considered fair = True.
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
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

class ThresholdOptimizer:
    def __init__(
        self,
        y_true,
        y_pred_proba,
        groups,
        initial_thresholds,
        learning_rate=1e-3,  # Smaller learning rate
        max_iterations=10000,
        relaxation_disparity=0.2,
        tolerance=1e-5,
        penalty=5,            # Reduced penalty term
        no_improvement_patience=500,  # If no improvement after these many iterations, reduce LR
        lr_reduction_factor=0.5,      # Reduce learning rate by this factor when no improvement
        delta=0.001            # Smaller delta for gradient approximation
    ):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.groups = groups
        self.group_indices = {group: (groups == group) for group in np.unique(groups)}

        # Initialize thresholds either from input or via grid search
        self.initial_thresholds = initial_thresholds.copy()
        self.thresholds = self.initial_thresholds.copy()

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.relaxation_disparity = relaxation_disparity
        self.tolerance = tolerance
        self.penalty = penalty
        self.no_improvement_patience = no_improvement_patience
        self.lr_reduction_factor = lr_reduction_factor
        self.delta = delta

        # Record history
        self.history = {group: {'accuracy': [], 'f1': [], 'threshold': []} for group in np.unique(groups)}

    def optimize(self):
        iteration = 0
        previous_thresholds = self.thresholds.copy()

        best_loss = np.inf
        no_improvement_counter = 0

        while iteration < self.max_iterations:
            confusion_matrix_df = pd.DataFrame()
            acc_dict, f1_dict = {}, {}

            # Calculate metrics for each group
            for group, indices in self.group_indices.items():
                group_y_true = self.y_true[indices]
                group_y_pred_proba = self.y_pred_proba[indices]
                threshold = self.thresholds[group]

                # Current predictions
                group_y_pred = group_y_pred_proba >= threshold
                acc = accuracy_score(group_y_true, group_y_pred)
                f1 = f1_score(group_y_true, group_y_pred, zero_division=1)

                acc_dict[group] = acc
                f1_dict[group] = f1

                # Record history
                self.history[group]['accuracy'].append(acc)
                self.history[group]['f1'].append(f1)
                self.history[group]['threshold'].append(threshold)

                confusion_matrix_df = self.update_confusion_matrix(
                    group, group_y_true, group_y_pred, confusion_matrix_df
                )

            # Compute the combined loss (negative accuracy average + penalty for very low F1)
            # You can customize this objective as needed.
            avg_acc = np.mean(list(acc_dict.values()))
            avg_f1 = np.mean(list(f1_dict.values()))
            loss = -avg_acc
            if avg_f1 < 0.5:  # example: penalize if avg_f1 < 0.5
                loss += self.penalty * (0.5 - avg_f1)

            # Update thresholds using gradient
            for group in self.group_indices.keys():
                indices = self.group_indices[group]
                group_y_true = self.y_true[indices]
                group_y_pred_proba = self.y_pred_proba[indices]
                threshold = self.thresholds[group]

                gradient = self.compute_gradient(group_y_true, group_y_pred_proba, threshold)

                # Update threshold
                new_threshold = threshold - self.learning_rate * gradient
                self.thresholds[group] = np.clip(new_threshold, 1e-7, 1 - 1e-7)

            # Check convergence and improvement
            max_threshold_change = max(
                abs(self.thresholds[group] - previous_thresholds[group]) for group in self.thresholds
            )

            # Track improvement
            if loss < best_loss:
                best_loss = loss
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            # Adaptive learning rate: if no improvement, reduce LR
            if no_improvement_counter > self.no_improvement_patience:
                self.learning_rate *= self.lr_reduction_factor
                no_improvement_counter = 0
                print(f"No improvement for {self.no_improvement_patience} iterations. "
                      f"Reducing learning rate to {self.learning_rate}.")

            # Check if thresholds have converged
            if max_threshold_change < self.tolerance:
                # Check if fairness conditions are met
                if self.check_fairness(confusion_matrix_df):
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

    def check_fairness(self, confusion_matrix_df):
        relaxation = self.relaxation_disparity
        fpr_values = confusion_matrix_df['FPR'].fillna(0).values
        tpr_values = confusion_matrix_df['TPR'].fillna(0).values
        ppr_values = confusion_matrix_df['PPR'].fillna(0).values

        fpr_disparity = fpr_values.max() - fpr_values.min()
        tpr_disparity = tpr_values.max() - tpr_values.min()
        ppr_disparity = ppr_values.max() - ppr_values.min()

        dp_condition = ppr_disparity <= relaxation
        eo_condition = (fpr_disparity <= relaxation) and (tpr_disparity <= relaxation)

        return dp_condition and eo_condition

    def compute_gradient(self, group_y_true, group_y_pred_proba, threshold):
        # Evaluate loss at threshold_plus
        threshold_plus = min(threshold + self.delta, 1)
        group_y_pred_plus = group_y_pred_proba >= threshold_plus
        acc_plus = accuracy_score(group_y_true, group_y_pred_plus)
        f1_plus = f1_score(group_y_true, group_y_pred_plus, zero_division=1)
        loss_plus = -acc_plus
        if f1_plus < 0.5:
            loss_plus += self.penalty * (0.5 - f1_plus)

        # Evaluate loss at threshold_minus
        threshold_minus = max(threshold - self.delta, 0)
        group_y_pred_minus = group_y_pred_proba >= threshold_minus
        acc_minus = accuracy_score(group_y_true, group_y_pred_minus)
        f1_minus = f1_score(group_y_true, group_y_pred_minus, zero_division=1)
        loss_minus = -acc_minus
        if f1_minus < 0.5:
            loss_minus += self.penalty * (0.5 - f1_minus)

        # Numerical gradient
        gradient = (loss_plus - loss_minus) / (2 * self.delta)
        return gradient

    def grid_search_thresholds(self, num=100):
        possible_thresholds = np.linspace(0, 1, num=num)
        best_thresholds = {}
        for group in self.group_indices.keys():
            best_acc = -np.inf
            best_threshold = 0.5
            group_indices = self.group_indices[group]
            group_y_true = self.y_true[group_indices]
            group_y_pred_proba = self.y_pred_proba[group_indices]
            for thr in possible_thresholds:
                group_y_pred = group_y_pred_proba >= thr
                acc = accuracy_score(group_y_true, group_y_pred)
                f1 = f1_score(group_y_true, group_y_pred, zero_division=1)
                # Example condition: choose threshold with best accuracy and decent F1
                if f1 >= 0.5 and acc > best_acc:
                    best_acc = acc
                    best_threshold = thr
            best_thresholds[group] = best_threshold
            print(f"Group: {group}, Best Threshold: {best_threshold:.4f}, Best Accuracy: {best_acc:.4f}")
        return best_thresholds


######################## RUN

# Load dataset
dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv")
df = dataset.sample(frac=1, random_state=42)

# Create groups based on length, formality, sentiment, personality
length_groups = pd.cut(
    df['text_length'].dropna(),
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values

formality_groups = pd.cut(
    df['formality'].dropna(),
    bins=[0, 50, np.inf],
    labels=['informal', 'formal']
).astype(str).values

sentiment_groups = df['sentiment_label'].fillna('neutral').astype(str).values
personality_groups = df['personality'].fillna('unknown').astype(str).values

# Combine into a single group label
groups = pd.Series([
    f"{length}_{formality}_{sentiment}_{personality}"
    for length, formality, sentiment, personality in zip(
        length_groups, formality_groups, sentiment_groups, personality_groups
    )
]).values

# Prepare true labels and predicted probabilities
y_true = df['AI_written'].apply(lambda x: 1 if x == 'AI' else 0).values
y_pred_proba = df['roberta_large_openai_detector_probability'].values

# Initial thresholds
initial_thresholds = {group: 0.5 for group in np.unique(groups)}

optimizer = ThresholdOptimizer(
    y_true=y_true,
    y_pred_proba=y_pred_proba,
    groups=groups,
    initial_thresholds=initial_thresholds,
    learning_rate=1e-3,
    max_iterations=5000,             # Allow more iterations for convergence
    relaxation_disparity=0.2,
    tolerance=1e-5, 
    penalty=5,
    no_improvement_patience=500,
    lr_reduction_factor=0.5,
    delta=0.001,
    # min_acc_threshold=0.5,
    # min_f1_threshold=0.5
)

thresholds, history = optimizer.optimize()

# If no change, try grid search
if all(np.isclose(thr, 0.5) for thr in thresholds.values()):
    print("Thresholds did not significantly change. Attempting grid search.")
    thresholds = optimizer.grid_search_thresholds()

# Print final thresholds
print("\nOptimized Thresholds:")
for group, threshold in thresholds.items():
    print(f"Group: {group}, Threshold: {threshold:.7f}")



##################################TEST

# need to apply the generated thresold to the test dataset
# Load test dataset
test_dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_t3_features.csv")

# Split test dataset by 'source'
unique_sources = test_dataset['source'].unique()

for source in unique_sources:
    source_dataset = test_dataset[test_dataset['source'] == source]

    # Split by different detector probabilities
    detector_probabilities = ['roberta_large_openai_detector_probability', 'radar_probability', 'roberta_base_openai_detector_probability']

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

        print(f"\nPerformance for Source: {source}, Detector: {detector}")
        print(f"Accuracy: {test_accuracy:.3f}")
        print(f"False Positive Rate (FPR): {test_fpr:.3f}")

        # store the printed things in txt and threshold for each
        with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results.txt", "a") as f:
            f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
            f.write(f"Accuracy: {test_accuracy:.3f}\n")
            f.write(f"False Positive Rate (FPR): {test_fpr:.3f}\n")
            f.write(f"Thresholds:\n")
            for group, threshold in thresholds.items():
                f.write(f"Group: {group}, Threshold: {threshold:.7f}\n")

        
        




