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
        min_acc_threshold=[0.5],      # Minimum required levels of Accuracy for each group.
        min_f1_threshold=[0.5],       # F1 is for stable performance. Minimum required levels of F1 score for each group.
        tolerance=1e-4,             # Convergence criterion (minimum change in thresholds between iterations).
        penalty=10,                 # Penalty applied if the F1 score is below the minimum threshold.
    ): 
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.groups = groups
        self.initial_thresholds = {group: initial_thresholds for group in np.unique(groups)}
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
        self.thresholds = {group: initial_thresholds for group in np.unique(groups)}
        self.initial_learning_rate = learning_rate  # Store initial learning rate
        self.delta_fairness = 0.0  #0.02
        self.delta_performance = 0.0  #0.2
        self.delta = False
        self.is_convergence = False
        


    # Optimizing thresholds to maximize metrics while maintaining fairness.
    def optimize(self):
    
        # Print the initial state once
        if not hasattr(self, 'printed_initial_state'):
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
            iterations += 1
            confusion_matrix_df = pd.DataFrame()
            acc_dict, f1_dict = {}, {}

            # Calculate metrics for each group
            # E.g.: group:   long_formal_NEGATIVE_extroversion
            #       indices: [False False False ... False False  True]
            for group, indices in self.group_indices.items():
                #print('='*500)
                #print(group)
                #print(self.min_acc_threshold[group])
                #print(self.min_f1_threshold[group])
                #print(self.thresholds[group])
                group_y_true = self.y_true[indices]
                group_y_pred_proba = self.y_pred_proba[indices]
                #threshold = self.thresholds[group]
                #print(self.y_true.shape)
                #print(group_y_true.shape)
                #print(self.min_acc_threshold[group])
                #print(self.min_f1_threshold[group])
                
                
                # Calculate current predictions
                group_y_pred = (group_y_pred_proba >= self.thresholds[group])

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
                    group_y_true, group_y_pred_proba, 
                    self.thresholds[group], self.min_acc_threshold[group], self.min_f1_threshold[group]
                )
                #print("gradient:",gradient)

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
                    print(f"Gradient = {gradient:.5f}, learning rate = {self.learning_rate}, self.thresholds[group] = {self.thresholds[group]}")
                    print(f"Confusion_matrix_df: \n{confusion_matrix_df}")
                    self.check_fairness(confusion_matrix_df, True)
                    self.check_performance_criteria(acc_dict, f1_dict, True)
                    print(f"self.delta_fairness: {self.delta_fairness }")
                    print(f"self.delta_performance: {self.delta_performance }")
                    print(f"self.delta: {self.delta }")
                    pass   
                    
            # Check convergence            
            threshold_changes = [abs(self.thresholds[group] - previous_thresholds[group]) for group in self.thresholds]
            max_threshold_change = max(threshold_changes)
            
            if iterations % 50 == 0:
                print(f"Current thresholds: {self.thresholds}, \nPrevious thresholds: {previous_thresholds}")
            
            #print(f"\nMax threshold change: {max_threshold_change}, Tolerance: {self.tolerance}")
                
            if max_threshold_change < self.tolerance:
                if self.check_fairness(confusion_matrix_df) and self.check_performance_criteria(acc_dict, f1_dict):
                    self.is_convergence = True
                    print(f"\nConverged after {iterations + 1} iterations.\n")
                    break

            # in case there is no change of the threshold
            #for group in self.thresholds:
                #if abs(self.thresholds[group] - previous_thresholds[group]) < self.tolerance:
                    #random_number = np.random.uniform(0, 1)
                    #self.thresholds[group] -= self.learning_rate * gradient * random_number
                    #self.thresholds[group] = np.clip(self.thresholds[group], 0.00005, 0.99995)
                    #self.thresholds[group] = self.initial_thresholds[group]                
            
            previous_thresholds = self.thresholds.copy()
            
            if any(value > 0.999 for value in self.thresholds.values()):
                #self.thresholds = self.initial_thresholds.copy()
                self.thresholds = {key: 0.01 for key in self.initial_thresholds}
                self.learning_rate -= 1*10**-1*self.initial_learning_rate #/ 2
                if self.learning_rate < 5*10**-6:
                    print("#-"*50)
                    self.initial_learning_rate = 1*10**-1*self.initial_learning_rate
                    self.learning_rate = self.initial_learning_rate
                print("="*500)
                print(f"self.thresholds: {self.thresholds}")
                print(f"self.learning_rate: {self.learning_rate}")                
            
            if iterations == self.max_iterations and self.delta == False:
                self.max_iterations *= 2
                self.delta_fairness = 0.02
                self.delta_performance = 0.2
                self.delta = True
                progress_bar.total = self.max_iterations
                progress_bar.desc = "Progress (to 2 * max iterations)"
           
            progress_bar.update(1)
                        
        progress_bar.close()

        return self.thresholds, iterations, self.learning_rate, acc_dict, f1_dict, self.is_convergence, self.delta


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
        
        if imprimir:
            print(f"PPR Disparity: {ppr_disparity} (<= {self.acceptable_ppr_disparity})")
            print(f"FPR Disparity: {fpr_disparity} (<= {self.acceptable_fpr_disparity})")
            print(f"TPR Disparity: {tpr_disparity} (<= {self.acceptable_tpr_disparity})")
            #if ppr_disparity <= self.acceptable_ppr_disparity and fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity:
            #    if ppr_disparity > 0.0 and fpr_disparity > 0.0 and tpr_disparity > 0.0:
            #        print("%___"*100)
            #        print(f"REACH FAIRNESS CRITERIA")
            #        print("%___"*100)
            #        with open(self.path, "a") as f:
            #            f.write(f'relaxation: {self.acceptable_fpr_disparity}')
            #            f.write(f'Learnig rate: {self.learning_rate}')
            #            f.write(f"PPR Disparity: {ppr_disparity} (<= {self.acceptable_ppr_disparity})\n")
            #            f.write(f"FPR Disparity: {fpr_disparity} (<= {self.acceptable_fpr_disparity})\n")
            #            f.write(f"TPR Disparity: {tpr_disparity} (<= {self.acceptable_tpr_disparity})\n")
            #            f.write(f'self.thresholds: {self.thresholds}')
            #        
            #        import time
            #        print("Esperando 10 minutos (600 segundos)...")
            #        time.sleep(60*10) 
                            
        # Demographic Parity (DP): ppr_disparity <= self.acceptable_ppr_disparity 
        # Equalized Odds (EO): fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity
        ppr_check = (ppr_disparity <= (self.acceptable_ppr_disparity + self.delta_fairness))
        fpr_check = (fpr_disparity <= (self.acceptable_fpr_disparity + self.delta_fairness))
        tpr_check = (tpr_disparity <= (self.acceptable_tpr_disparity + self.delta_fairness))
        #if ppr_disparity <= self.acceptable_ppr_disparity and fpr_disparity <= self.acceptable_fpr_disparity and tpr_disparity <= self.acceptable_tpr_disparity:
        if ppr_check and fpr_check and tpr_check:
            return True
        else:
            return False
        
        
    def check_performance_criteria(self, acc_dict, f1_dict, imprimir = False):
        
        acc_check = all(acc_dict[group] >= (self.min_acc_threshold[group] - self.delta_performance) for group in acc_dict)
        f1_check = all(f1_dict[group] >= (self.min_f1_threshold[group] - self.delta_performance) for group in f1_dict)
        
        if imprimir:
            print("\nPerformance criteria met:")
            print("Accuracy thresholds:")
            for group, acc in acc_dict.items():
                print(f"Group {group}: Accuracy = {acc:.4f} (>= min Threshold = {self.min_acc_threshold[group]})")
            print("F1 score thresholds:")
            for group, f1 in f1_dict.items():
                print(f"Group {group}: F1 Score = {f1:.4f} (>= min Threshold = {self.min_f1_threshold[group]})")
 
        if acc_check and f1_check:
            return True
        else:
            return False


    def compute_gradient(self, group_y_true, group_y_pred_proba, min_acc_threshold, threshold, min_f1_threshold):
        # Calculate current predictions
        group_y_pred = (group_y_pred_proba >= threshold)

        # Calculate accuracy and F1 score
        acc = accuracy_score(group_y_true, group_y_pred)
        f1 = f1_score(group_y_true, group_y_pred, zero_division=1)

        # Compute loss: negative accuracy (to maximize accuracy)
        loss = -acc
        #loss = 0
            
        if acc < min_acc_threshold:
            loss += self.penalty * (min_acc_threshold - acc)
            
        # Add penalty if F1 score is below the minimum threshold
        if f1 < min_f1_threshold:
            loss += self.penalty * (min_f1_threshold - f1)

        #gradient = loss


        # Adjust delta
        #delta = 0.01 #0.01  # Increased delta

        #threshold_plus = min(threshold + delta, 1)
        #threshold_minus = max(threshold - delta, 0)
        
        # Loss at threshold_plus
        #group_y_pred_plus = group_y_pred_proba >= threshold_plus
        #acc_plus = accuracy_score(group_y_true, group_y_pred_plus)
        #f1_plus = f1_score(group_y_true, group_y_pred_plus, zero_division=1)
        #loss_plus = -acc_plus
        #if f1_plus < self.min_f1_threshold:
        #    loss_plus += self.penalty * (self.min_f1_threshold - f1_plus)

        #gradient = loss_plus
        #print(f"Gradient: {gradient}, Loss: {loss}, Loss_minus_plus: {loss_minus_plus}, Delta: {delta}")
        #print(f"Accuracy: {acc}, F1 Score: {f1}")
        #print(f"Accuracy (minus/plus): {acc_minus_plus}, F1 Score (minus/plus): {f1_minus_plus}\n")

        return loss #loss_plus #loss #gradient