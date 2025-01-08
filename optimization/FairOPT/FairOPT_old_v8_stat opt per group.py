import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


class ThresholdOptimizer:
    def __init__(
        self,
        y_true,                     # Vector of true labels (1 or 0).
        y_pred_proba,               # Probabilities predicted by the model (​​between 0 and 1).
        groups,                     # Identifies the data groups to assess fairness.
        initial_thresholds,         # For each group, which will convert probabilities into binary predictions.
        group_indices,             # Dictionary mapping each group to a boolean vector indicating which samples belong to that group.
        learning_rate=10**-3,       # Adjusted learning rate. For threshold adjustment.
        max_iterations=10**5,       # Maximum number of iterations.
        acceptable_disparity=0.2,   # Acceptable disparity between groups. Maximum allowed level of disparity between group metrics.
        min_acc_threshold=0.5,      # Minimum required levels of Accuracy for each group.
        min_f1_threshold=0.5,       # F1 is for stable performance. Minimum required levels of F1 score for each group.
        tolerance=1e-4,             # Convergence criterion (minimum change in thresholds between iterations).
        penalty=10                  # Penalty applied if the F1 score is below the minimum threshold.
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
        self.group_indices = group_indices#{group: (groups == group) for group in np.unique(groups)}  # Dictionary mapping each group to a boolean vector indicating which samples belong to that group.
        self.history = {group: {'accuracy': [], 'f1': [], 'threshold': []} for group in np.unique(groups)}  # Dictionary to store the history of accuracy, F1 score, and thresholds for each group.
        self.thresholds = self.initial_thresholds.copy()
        self.initial_learning_rate = learning_rate  # Store initial learning rate
        self.aux = True
        self.temp_value = {key: self.initial_thresholds[key] for key in self.groups}



    # Optimizing thresholds to maximize metrics while maintaining fairness.
    def optimize(self):
        
        # Print the initial state once
        if not hasattr(self, 'printed_initial_state'):
            print("temp_value: ", self.temp_value)
            print("Initial thresholds:", self.initial_thresholds)
            print("Learning rate:", self.learning_rate)
            print("Max iterations:", self.max_iterations)
            print("Acceptable disparity:", self.acceptable_ppr_disparity)
            print("Min accuracy threshold:", self.min_acc_threshold)
            print("Min F1 threshold:", self.min_f1_threshold)
            print("Tolerance:", self.tolerance)
            print("Penalty:", self.penalty)
            self.printed_initial_state = True
        
        
        iterations = 0
        previous_thresholds = self.thresholds.copy()
        
        #Create a progress bar
        print(f"Optimization progress until convergence or maximum iterations reached ({self.max_iterations:,} iterations). Tolerance: {self.tolerance}")
        progress_bar = tqdm(total=self.max_iterations, desc="Progress (to max iterations)")

        while iterations < self.max_iterations:  # Loop until convergence or maximum iterations reached
            confusion_matrix_df = pd.DataFrame()
            acc_dict, f1_dict = {}, {}

            # Calculate metrics for each group
            # E.g.: group:   long_formal_NEGATIVE_extroversion
            #       indices: [False False False ... False False  True]
            
            for group, indices in self.group_indices.items():
                group_y_true = self.y_true[indices]
                group_y_pred_proba = self.y_pred_proba[indices]
                threshold = self.thresholds[group]
                
                # Calculate current predictions
                group_y_pred = group_y_pred_proba >= threshold

                # Calculate accuracy and F1 score
                acc = accuracy_score(group_y_true, group_y_pred)
                f1 = f1_score(group_y_true, group_y_pred, zero_division=1) # zero_division=1 sets the metric to 1 if a division by zero occurs during calculation.
                acc_dict[group] = acc
                f1_dict[group] = f1

                # Record the metrics and update confusion matrix
                #self.history[group]['accuracy'].append(acc)
                #self.history[group]['f1'].append(f1)
                #self.history[group]['threshold'].append(threshold)
                confusion_matrix_df = self.update_confusion_matrix(
                    group, group_y_true, group_y_pred, confusion_matrix_df
                )

                gradient = self.compute_gradient(
                    group_y_true, group_y_pred_proba, threshold
                )

                # Check if gradient is effectively zero
                #if abs(gradient) < 1e-7:
                    #print(f"Iteration {iteration}, Group {group}: Gradient is zero, adjusting delta or learning rate.")
                    # Optionally increase delta or adjust learning rate here if needed

                # Update threshold
                #self.thresholds[group] = threshold - self.learning_rate * gradient
                
                self.thresholds[group] -= self.learning_rate * gradient
                self.thresholds[group] = np.clip(self.thresholds[group], 0.00005, 0.99995) # Ensures the group's thresholds stay within the range
                

            
                if iterations % 50 == 0:# and group == "medium":                    
                    print(f"\nGroup <{group}>:")
                    print(f"#1s: {np.sum(group_y_pred):,}, #0s: {len(group_y_pred) - np.sum(group_y_pred):,}")
                    print(f"ACC: {acc}, F1: {f1}")
                    print(f"Gradient = {gradient:.5f}, threshold = {threshold}, [t] = {self.thresholds[group]}")
                    print(f"Confusion_matrix_df: \n{confusion_matrix_df}")
                    self.check_fairness(confusion_matrix_df, True)
                    self.check_performance_criteria(acc_dict, f1_dict, True)
                    pass   
                    
            # Check convergence            
            threshold_changes = [abs(self.thresholds[group] - previous_thresholds[group]) for group in self.thresholds]
            max_threshold_change = max(threshold_changes)
            
            if iterations % 50 == 0:
                print(f"Current thresholds: {self.thresholds}, \nPrevious thresholds: {previous_thresholds}")
            
            #print(f"\nMax threshold change: {max_threshold_change}, Tolerance: {self.tolerance}")
            
            #fairness, excluded_rows = self.check_fairness(confusion_matrix_df)                 
            #for group, exclude in excluded_rows.items():
            #    if not exclude:  # Only update if the value in excluded_rows is False
            #        self.thresholds[group] -= self.learning_rate * gradient
            #        self.thresholds[group] = np.clip(self.thresholds[group], 0.00005, 0.99995) # Ensures the group's thresholds stay within the range


            if max_threshold_change < self.tolerance:
                fairness, excluded_rows = self.check_fairness(confusion_matrix_df) 
                if fairness and self.check_performance_criteria(acc_dict, f1_dict):
                    print(f"\nConverged after {iterations + 1} iterations.\n")
                    break

            fairness, excluded_rows = self.check_fairness(confusion_matrix_df)                 
            if any(excluded_rows.values()) and self.aux: #At least one value is True.
                print("="*500)
                print(excluded_rows)
                print(self.thresholds)
                for group, exclude in excluded_rows.items():
                    print(group, exclude)
                    if not exclude:  # Only update if the value in excluded_rows is False
                        self.thresholds[group] = self.initial_thresholds[group]    
                        print("inside: ", group)
                    if exclude: 
                        self.temp_value[group] = self.thresholds[group]
                self.aux = False
                print(self.thresholds)
                print("="*500)
                    
   
            previous_thresholds = self.thresholds.copy()
            iterations += 1
            
            progress_bar.update(1)
                        
        progress_bar.close()

        return self.thresholds, (iterations+1) # add 1 because it started at 0.


    def update_confusion_matrix(self, group, y_true, y_pred, confusion_matrix_df):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        confusion_matrix_df.loc[group, 'True Positives'] = tp
        confusion_matrix_df.loc[group, 'True Negatives'] = tn
        confusion_matrix_df.loc[group, 'False Positives'] = fp
        confusion_matrix_df.loc[group, 'False Negatives'] = fn

        confusion_matrix_df.loc[group, 'PPR'] = (tp + fp) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0 # PPR: Positive Prediction Rate
        confusion_matrix_df.loc[group, 'FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        confusion_matrix_df.loc[group, 'TPR'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        return confusion_matrix_df


    # check fairness 
    def check_fairness(self, confusion_matrix_df, imprimir = False):        
        
        # Define acceptable disparity thresholds
        ppr_values = confusion_matrix_df['PPR'].fillna(0).values
        fpr_values = confusion_matrix_df['FPR'].fillna(0).values
        tpr_values = confusion_matrix_df['TPR'].fillna(0).values

        ppr_disparity = ppr_values.max() - ppr_values.min()
        fpr_disparity = fpr_values.max() - fpr_values.min()
        tpr_disparity = tpr_values.max() - tpr_values.min()
        
        
        # Identify indices for min and max values of PPR, FPR, and TPR
        ppr_min_idx, ppr_max_idx = np.argmin(ppr_values), np.argmax(ppr_values)
        fpr_min_idx, fpr_max_idx = np.argmin(fpr_values), np.argmax(fpr_values)
        tpr_min_idx, tpr_max_idx = np.argmin(tpr_values), np.argmax(tpr_values)

        # Get row names (index labels)
        row_names = confusion_matrix_df.index.tolist()

        # Create a dictionary with all rows marked as True
        excluded_rows = {name: True for name in row_names}

        # Mark rows with min or max values as False
        indices_to_include = {ppr_min_idx, ppr_max_idx, fpr_min_idx, fpr_max_idx, tpr_min_idx, tpr_max_idx}
        for idx in indices_to_include:
            excluded_rows[row_names[idx]] = False
        
        
        if imprimir:
            #print(confusion_matrix_df)
            print(excluded_rows)
            print(indices_to_include)
            print(np.argmin(ppr_values), np.argmax(ppr_values))
            print(np.argmin(fpr_values), np.argmax(fpr_values))
            print(np.argmin(tpr_values), np.argmax(tpr_values))
            print(f"PPR Disparity: {ppr_disparity} (<= {self.acceptable_ppr_disparity})")
            print(f"FPR Disparity: {fpr_disparity} (<= {self.acceptable_fpr_disparity})")
            print(f"TPR Disparity: {tpr_disparity} (<= {self.acceptable_tpr_disparity})")
                
        # Demographic Parity (DP): ppr_disparity <= self.acceptable_ppr_disparity 
        # Equalized Odds (EO): fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity
        if ppr_disparity <= self.acceptable_ppr_disparity and fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity:
            return True, excluded_rows
        else:
            return False, excluded_rows
        
        
    def check_performance_criteria(self, acc_dict, f1_dict, imprimir = False):
        if imprimir:
            print("\nPerformance criteria met:")
            print("Accuracy thresholds:")
            for group, acc in acc_dict.items():
                print(f"Group {group}: Accuracy = {acc:.4f} (>= min Threshold = {self.min_acc_threshold})")
            print("F1 score thresholds:")
            for group, f1 in f1_dict.items():
                print(f"Group {group}: F1 Score = {f1:.4f} (>= min Threshold = {self.min_f1_threshold})")
 
        if all(acc >= self.min_acc_threshold for acc in acc_dict.values()) and all(f1 >= self.min_f1_threshold for f1 in f1_dict.values()):
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
        delta = 0.01 #0.01  # Increased delta

        threshold_plus = min(threshold + delta, 1)
        threshold_minus = max(threshold - delta, 0)
        
        #print(f"threshold: {threshold}, threshold_plus: {threshold_plus}, threshold_minus: {threshold_minus}")

        # Loss at threshold_minus and threshold_plus
        #group_y_pred_minus_plus = (group_y_pred_proba >= threshold_minus) & (group_y_pred_proba <= threshold_plus)
        
        # Count the number of 1s and 0s in group_y_pred_minus_plus
        #num_ones = np.sum(group_y_pred_minus_plus)
        #num_zeros = len(group_y_pred_minus_plus) - num_ones
        #print(f"group_y_pred_minus_plus - # 1s: {num_ones}")
        #print(f"group_y_pred_minus_plus - # 0s: {num_zeros}")
        #print(f"group_y_true - min: {np.min(group_y_true)}, max: {np.max(group_y_true)}")
        
        #acc_minus_plus = accuracy_score(group_y_true, group_y_pred_minus_plus)
        #f1_minus_plus = f1_score(group_y_true, group_y_pred_minus_plus, zero_division=1)
        #loss_minus_plus = -acc_minus_plus
        #if f1_minus_plus < self.min_f1_threshold:
        #    loss_minus_plus += self.penalty * (self.min_f1_threshold - f1_minus_plus)

        # Loss at threshold_plus
        group_y_pred_plus = group_y_pred_proba >= threshold_plus
        acc_plus = accuracy_score(group_y_true, group_y_pred_plus)
        f1_plus = f1_score(group_y_true, group_y_pred_plus, zero_division=1)
        loss_plus = -acc_plus
        if f1_plus < self.min_f1_threshold:
            loss_plus += self.penalty * (self.min_f1_threshold - f1_plus)
        #print(f"group_y_pred_plus - # 1s: {num_ones}")
        #print(f"group_y_pred_plus - # 0s: {num_zeros}\n")
            
        #print(f"Acc: {acc}, acc_plus {acc_plus}")   
        # Compute numerical gradient
        #gradient = (loss_minus_plus) / (2 * delta)
        ##gradient = loss_plus # / delta
        gradient = loss_plus
        #print(f"Gradient: {gradient}, Loss: {loss}, Loss_minus_plus: {loss_minus_plus}, Delta: {delta}")
        #print(f"Accuracy: {acc}, F1 Score: {f1}")
        #print(f"Accuracy (minus/plus): {acc_minus_plus}, F1 Score (minus/plus): {f1_minus_plus}\n")

        return gradient