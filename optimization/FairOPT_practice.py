import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# Interpreting equaltion
def equation_fairness(Prob_a,Prob_b):
    rule = 0.8 # threshold of 0.8 is based on 80% rule. # can be changed
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
def Demographic_Disparity(confusion_matrix_df):
    # Assuming confusion_matrix_df has a 'Group' index and a column 'FPR' for False Positive Rate

    # Iterate through pairs of groups to compare their FPRs
    groups = confusion_matrix_df.index
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_a, group_b = groups[i], groups[j]
            fpr_a = confusion_matrix_df.loc[group_a, 'FPR']
            fpr_b = confusion_matrix_df.loc[group_b, 'FPR']

            # Evaluate fairness between each pair of groups
            fair = equation_fairness(fpr_a, fpr_b)
            print(f"Fairness between Group {group_a} and Group {group_b}: {fair}")
    return fair


class ThresholdOptimizer:
    def __init__(
        self,
        y_true,
        y_pred_proba,
        groups,
        initial_thresholds,
        learning_rate=10**-7,   # Adjusted learning rate
        max_iterations=10**7,
        acceptable_fpr_disparity=0.2,
        acceptable_tpr_disparity=0.2,
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
        self.acceptable_fpr_disparity = acceptable_fpr_disparity
        self.acceptable_tpr_disparity = acceptable_tpr_disparity
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
                self.thresholds[group] = np.clip(self.thresholds[group], 0, 1)

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

        confusion_matrix_df.loc[group, 'True Positives'] = tp
        confusion_matrix_df.loc[group, 'True Negatives'] = tn
        confusion_matrix_df.loc[group, 'False Positives'] = fp
        confusion_matrix_df.loc[group, 'False Negatives'] = fn

        confusion_matrix_df.loc[group, 'FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        confusion_matrix_df.loc[group, 'TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        return confusion_matrix_df

    # check fairness 
    def check_fairness(self, confusion_matrix_df):        
        
        # Define acceptable disparity thresholds
        fpr_values = confusion_matrix_df['FPR'].fillna(0).values
        tpr_values = confusion_matrix_df['TPR'].fillna(0).values

        fpr_disparity = fpr_values.max() - fpr_values.min()
        tpr_disparity = tpr_values.max() - tpr_values.min()

        if fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity:
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
        print(f"Group Threshold: {threshold:.2f}, Delta: {delta}, Changes +delta: {changes_plus}, Changes -delta: {changes_minus}")

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

# import df
dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_Mage_d3.csv")

#split by train and tesxt <- change this later
df = dataset.sample(frac=0.8, random_state=42)
df_test = dataset.drop(df.index)



# Create group labels
length_groups = pd.cut(
    df['num_chars'],
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values  # Length-based groups

edu_groups = df['educational_level'].astype(str).values  # Educational level groups
sentiment_groups = df['sentiment_label'].astype(str).values  # Sentiment groups

# Combine groups into a single group label
groups = pd.Series([
    f"{length}_{edu}_{sent}"
    for length, edu, sent in zip(length_groups, edu_groups, sentiment_groups)
]).values

# Prepare true labels and predicted probabilities
y_true = df['AI_written'].apply(lambda x: 1 if x == 'AI' else 0).values  # True labels
y_pred_proba = df['roberta_base_openai_detector_probability'].values     # Predicted probabilities

# Initial thresholds (set to 0.5 for all groups)
initial_thresholds = {group: 0.5 for group in np.unique(groups)}

# Create an instance of ThresholdOptimizer
optimizer = ThresholdOptimizer(
    y_true,
    y_pred_proba,
    groups,
    initial_thresholds,
    learning_rate=10**-4,
    max_iterations=10**3,
    acceptable_fpr_disparity=0.2,  # Adjust based on your fairness criteria
    acceptable_tpr_disparity=0.2,  # Adjust accordingly
    min_acc_threshold=0.5,         # Set realistic minimum accuracy
    min_f1_threshold=0.5           # Set realistic minimum F1 score
)

# Optimize thresholds using gradient-based method
thresholds, history = optimizer.optimize()

# If thresholds still do not change, use grid search as an alternative
if all(threshold == 0.5 for threshold in thresholds.values()):
    print("Thresholds did not change using gradient descent. Switching to grid search.")
    thresholds = optimizer.grid_search_thresholds()

# View optimized thresholds
print("\nOptimized Thresholds:")
for group, threshold in thresholds.items():
    print(f"Group: {group}, Threshold: {threshold:.4f}")
