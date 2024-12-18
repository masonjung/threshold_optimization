# Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import cvxpy as cp

# Helper function for fairness evaluation
def equation_fairness(prob_a, prob_b, rule=0.8):
    greater = max(prob_a, prob_b)
    smaller = min(prob_a, prob_b)
    return (smaller / greater) > rule


class ConvexThresholdOptimizer:
    def __init__(self, y_true, y_pred_proba, groups, fairness_rule=0.8):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.groups = groups
        self.unique_groups = np.unique(groups)
        self.fairness_rule = fairness_rule

    def optimize(self):
        # Create CVXPY variables for thresholds
        thresholds = {group: cp.Variable() for group in self.unique_groups}

        # Create binary variables for predictions per group
        prediction_vars = {}
        group_indices_map = {}
        # Precompute Pos and Neg for each group (counts of actual positives/negatives)
        Pos = {}
        Neg = {}

        for group in self.unique_groups:
            idx = np.where(self.groups == group)[0]
            group_indices_map[group] = idx
            n = len(idx)
            prediction_vars[group] = cp.Variable(n, boolean=True)

            group_y_true = self.y_true[idx]
            Pos[group] = np.sum(group_y_true) + 1e-10  # Avoid division by zero
            Neg[group] = np.sum(1 - group_y_true) + 1e-10

        # Objective: Maximize accuracy
        objective_terms = []
        constraints = []

        # Constraints for threshold ranges
        for group in self.unique_groups:
            constraints += [thresholds[group] >= 0, thresholds[group] <= 1]

        # Build constraints for predictions
        # z[i] should be 1 if y_pred_proba[i] >= threshold[group], else 0
        # Using the big-M approach described above:
        for group in self.unique_groups:
            idx = group_indices_map[group]
            group_y_true = self.y_true[idx]
            group_y_pred_proba = self.y_pred_proba[idx]
            z = prediction_vars[group]
            t = thresholds[group]

            # For each sample, enforce z[i]:
            # group_y_pred_proba[i] - t <= z[i]
            # t - group_y_pred_proba[i] <= 1 - z[i]
            constraints += [group_y_pred_proba[i] - t <= z[i] for i in range(len(idx))]
            constraints += [t - group_y_pred_proba[i] <= 1 - z[i] for i in range(len(idx))]

            # Accuracy contributions:
            # correct_predictions = sum(y_true_i * z[i]) + sum((1-y_true_i)*(1-z[i]))
            correct_predictions = cp.sum(cp.multiply(group_y_true, z)) + cp.sum(cp.multiply(1 - group_y_true, 1 - z))
            objective_terms.append(correct_predictions)

        # Fairness constraints
        # We need to ensure the TPR and FPR differences across groups are constrained.
        # tpr_group = tp/Pos[group], fpr_group = fp/Neg[group]
        # tp = sum(y_true_i*z[i]), fp = sum((1-y_true_i)*z[i])
        # Add constraints for all group pairs
        for i, g1 in enumerate(self.unique_groups):
            for g2 in self.unique_groups[i+1:]:
                z1 = prediction_vars[g1]
                z2 = prediction_vars[g2]

                idx1 = group_indices_map[g1]
                idx2 = group_indices_map[g2]

                y1 = self.y_true[idx1]
                y2 = self.y_true[idx2]

                tp1 = cp.sum(cp.multiply(y1, z1))
                fp1 = cp.sum(cp.multiply(1 - y1, z1))

                tp2 = cp.sum(cp.multiply(y2, z2))
                fp2 = cp.sum(cp.multiply(1 - y2, z2))

                tpr1 = tp1 / Pos[g1]
                tpr2 = tp2 / Pos[g2]
                fpr1 = fp1 / Neg[g1]
                fpr2 = fp2 / Neg[g2]

                # |tpr1 - tpr2| <= 0.2
                constraints += [cp.abs(tpr1 - tpr2) <= 0.2]
                # |fpr1 - fpr2| <= 0.2
                constraints += [cp.abs(fpr1 - fpr2) <= 0.2]

        # Define objective
        objective = cp.Maximize(cp.sum(objective_terms))

        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        # Use a MIP solver. For example, CBC:
        problem.solve(solver=cp.CBC)  # or GUROBI if available

        # Store optimized thresholds
        self.optimized_thresholds = {group: thresholds[group].value for group in self.unique_groups}
        return self.optimized_thresholds


# Load dataset
dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv")

# Prepare groups
length_groups = pd.cut(
    dataset['text_length'].dropna(),
    bins=[0, 1000, 2500, np.inf],
    labels=['short', 'medium', 'long']
).astype(str).values

formality_groups = pd.cut(
    dataset['formality'].dropna(),
    bins=[0, 50, np.inf],
    labels=['informal', 'formal']
).astype(str).values

groups = pd.Series([
    f"{length}_{formality}"
    for length, formality in zip(length_groups, formality_groups)
]).values

# Prepare true labels and predicted probabilities
y_true = dataset['AI_written'].apply(lambda x: 1 if x == 'AI' else 0).values
y_pred_proba = dataset['roberta_large_openai_detector_probability'].values

# Optimize thresholds
optimizer = ConvexThresholdOptimizer(y_true, y_pred_proba, groups)
optimized_thresholds = optimizer.optimize()

# Print results
print("\nOptimized Thresholds:")
for group, threshold in optimized_thresholds.items():
    print(f"Group: {group}, Optimized Threshold: {threshold:.4f}")

thresholds = optimized_thresholds

##################################TEST

# Load test dataset
test_dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_features.csv")

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

        test_formality_groups = pd.cut(
            source_dataset['formality'],
            bins=[0, 50, np.inf],
            labels=['informal', 'formal']
        ).astype(str).values

        test_sentiment_groups = source_dataset['sentiment_label'].astype(str).values
        test_personality_groups = source_dataset['personality'].astype(str).values

        # Combine groups into a single group label for test dataset
        test_groups = pd.Series([
            f"{length}_{formality}_{sentiment}_{personality}"
            for length, formality, sentiment, personality in zip(
                test_length_groups, test_formality_groups, test_sentiment_groups, test_personality_groups
            )
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

        # Calculate performance metrics
        test_accuracy = accuracy_score(test_y_true, test_y_pred)
        test_fpr = np.sum((test_y_pred == 1) & (test_y_true == 0)) / np.sum(test_y_true == 0) if np.sum(test_y_true == 0) > 0 else 0

        print(f"\nPerformance for Source: {source}, Detector: {detector}")
        print(f"Accuracy: {test_accuracy:.4f}")
        print(f"False Positive Rate (FPR): {test_fpr:.4f}")

        # Store results in a txt file
        with open("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\results.txt", "a") as f:
            f.write(f"\nPerformance for Source: {source}, Detector: {detector}\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"False Positive Rate (FPR): {test_fpr:.4f}\n")
            f.write(f"Thresholds:\n")
            for group, threshold in thresholds.items():
                f.write(f"Group: {group}, Threshold: {threshold:.7f}\n")
