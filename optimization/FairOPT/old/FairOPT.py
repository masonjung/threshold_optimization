#import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
#from tqdm import tqdm
import time

class ConvergenceReached(Exception):
    pass


class ThresholdOptimizer:
    def __init__(
        self,
        y_true,                    # Vector of true labels (1 or 0).
        y_pred_proba,              # Probabilities predicted by the model (​​between 0 and 1).
        
        group_indices,             # Dictionary mapping each group to a boolean vector indicating which samples belong to that group.
        min_acc,                   # Minimum required levels of Accuracy for each group.
        min_f1,                    # Minimum required levels of F1 score for each group.
        min_disparity,             # Acceptable disparity between groups.
        
        learning_rate = 1e-3,      # Adjusted learning rate. For threshold adjustment.
        penalty=10,                # Penalty applied if the performance metrics score (acc and f1) is below the minimum threshold.
        max_iteration_minimize=10  # Maximum number of iterations of the "minimize" function, which objective is to minimize the loss function.
    ): 
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        
        self.group_indices = group_indices
        """
        # self.delta_performance = 0.1 convergence until disparity = 0.3
        # self.delta_performance = 0.2 convergence until disparity = 0.25
        # self.delta_performance = 0.3 convergence until disparity = 0.21 
        # self.delta_performance = 1.0 convergence until disparity = 0.20
        # change thresholds: 0.19:
            self.min_threshold = 0.0 #0.00005; since 0.19: 0
            self.max_threshold = 1.0 #0.99995: since 0.19: 1
        # For lower convergence <= 0.18, don't consider this and everything will be categorized as negative with ppr = 0.
            acc_check = acc_check and any( (acc_dict[group] >0) for group in acc_dict)
            f1_check = f1_check and any( (f1_dict[group] >0) for group in f1_dict)
        """
        
        self.min_threshold = 0.00005 #0.00005; since 0.19: 0
        self.max_threshold = 0.99995 #0.99995: since 0.19: 1
        
        self.min_disparity = min_disparity
        
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.max_iteration_minimize = max_iteration_minimize
        
        self.thresholds = {key: 1.0 for key in self.group_indices}
        self.group_losses = {key: 0.0 for key in self.group_indices}
        self.is_convergence = False
        self.thresholds_opt = {key: None for key in self.group_indices}
        self.acc_opt = 0

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
    
    
    def loss_function(self, thresholds):
        if not hasattr(self, 'first_minimize'):
            self.thresholds = {key: thresholds[i] for i, key in enumerate(self.group_indices.keys())}
            self.first_minimize = True
        
        #print(self.thresholds)
        confusion_matrix_df = pd.DataFrame()
        acc_dict, f1_dict, ppr_dict = {}, {}, {}
        total_loss = 0
        final_y_pred = np.zeros_like(self.y_true) 

        for group, indices in self.group_indices.items():
            group_y_true = self.y_true[indices]
            group_y_pred_proba = self.y_pred_proba[indices]

            group_y_pred = (group_y_pred_proba >= self.thresholds[group])  # Binary predictions
            final_y_pred[indices] = group_y_pred
            
            acc = accuracy_score(group_y_true, group_y_pred)
            f1 = f1_score(group_y_true, group_y_pred, zero_division=1)
            acc_dict[group] = acc
            f1_dict[group] = f1
            
            confusion_matrix_df = self.update_confusion_matrix(
                    group, group_y_true, group_y_pred, confusion_matrix_df
                )
            ppr_values = confusion_matrix_df['PPR'].fillna(0).values
            ppr_disparity = ppr_values.max() - ppr_values.min()
            ppr_dict[group] = ppr_disparity
            
            # Compute loss
            group_loss = 1 - acc   # Minimize loss based on accuracy
            group_loss += (1 - f1)   # Minimize loss based on F1
            if ppr_disparity > self.min_disparity:
                group_loss += (self.penalty * (self.min_disparity - ppr_disparity) ** 2)  # L2 penalty
                
            total_loss += group_loss  # Accumulate the total loss
            if group_loss <= self.group_losses[group]:
                factor = 1
            else: 
                factor = -0.5
            self.group_losses[group] = group_loss
            
            self.thresholds[group] -= self.learning_rate * group_loss * factor  # Update the threshold
            self.thresholds[group] = np.clip(self.thresholds[group], self.min_threshold, self.max_threshold)  # Ensure thresholds stay within bounds
                           
        acc = accuracy_score(self.y_true, final_y_pred)
        f1 = f1_score(self.y_true, final_y_pred, zero_division=1)
        
        confusion_matrix_df = self.update_confusion_matrix(
            group, group_y_true, group_y_pred, confusion_matrix_df
        )
        ppr_values = confusion_matrix_df['PPR'].fillna(0).values
        ppr_disparity = ppr_values.max() - ppr_values.min()
        ppr_check  = ( ppr_disparity  <= self.min_disparity )
        
        
        if ppr_check == True:
            self.is_convergence = True
            #print("Convergence reached")
            #print("|"*100)
            ##stop because the convergence is reached
            #print(f'acc: {acc} - f1: {f1} - ppr_disparity: {ppr_disparity}')
            ##raise ConvergenceReached  # Raise the exception to stop optimization
            
            if self.acc_opt < acc:
                self.acc_opt = acc
                print(f'acc_opt: {self.acc_opt}, f1: {f1} , ppr_disparity: {ppr_disparity}')
                self.thresholds_opt = self.thresholds.copy()
               
        return total_loss


    
    def optimize(self):
        # Call the optimizer
        try:
            for i in range(self.max_iteration_minimize+1):
                print(i)
                start_time = time.time()
                result = minimize(
                    fun     = self.loss_function, 
                    x0      = list(self.thresholds.values()), # list_thresholds,  # Initial guess for thresholds
                    bounds  = [(0, 1)] * len(self.thresholds),  # Ensure thresholds stay within bounds
                    method  = 'L-BFGS-B'  # You can choose other methods if needed
                    #options={'maxiter': 20}  # Control max calls to "fun"
                )
                end_time = time.time()
                execution_time = end_time - start_time
                minutes = execution_time // 60
                seconds = execution_time % 60
                print (f"Execution time: {minutes} minutes and {seconds} seconds", end="\n\n")
            # Update the optimized thresholds
            #self.thresholds = result.x  # Set the optimized thresholds
            print("Optimized thresholds:", self.thresholds_opt )
            return self.is_convergence, self.thresholds_opt # Return the optimized thresholds
        except ConvergenceReached:
            print("Optimization stopped early due to convergence.")
            return self.is_convergence, self.thresholds_opt 