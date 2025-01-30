#import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
#from tqdm import tqdm
import time

class ThresholdOptimizer:
    def __init__(
        self,
        y_true,                    # Vector of true labels (1 or 0).
        y_pred_proba,              # Probabilities predicted by the model (​​between 0 and 1).
        
        group_indices,             # Dictionary mapping each group to a boolean vector indicating which samples belong to that group.
        acceptable_disparities,    # List of disparities to test.
        
        learning_rate = 1e-3,      # Adjusted learning rate. For threshold adjustment.
        penalty=10,                # Penalty applied if the performance metrics score (acc and f1) is below the minimum threshold.
        max_iteration_minimize=10  # Maximum number of iterations of the "minimize" function, which objective is to minimize the loss function.
    ): 
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        
        self.group_indices = group_indices        
        self.min_threshold = 0.00005
        self.max_threshold = 0.99995
        
        self.max_disparity = 1.0
        self.acceptable_disparities = sorted(acceptable_disparities, reverse=True)
        
        self.learning_rate = learning_rate
        self.penalty = penalty
        self.max_iteration_minimize = max_iteration_minimize
        
        self.thresholds = {key: 1.0 for key in self.group_indices}
        self.group_losses = {key: 0.0 for key in self.group_indices}
        self.best_results = {d: {'acc_opt': 0, 'f1': 0, 'thresholds_opt': {} } for d in self.acceptable_disparities}
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
        
        for d in self.acceptable_disparities:
            if ppr_disparity <= d and acc > self.best_results[d]['acc_opt']:
                self.best_results[d]['acc_opt'] = acc
                self.best_results[d]['f1'] = f1
                self.best_results[d]['thresholds_opt'] = self.thresholds.copy()
                #print(f'Updated best results for disparity {d}: acc {acc}, f1 {f1}, ppr {ppr_disparity}')
        
        return total_loss


    def optimize(self):
        for i in range(self.max_iteration_minimize + 1):
            minimize(
                fun=self.loss_function,
                x0=list(self.thresholds.values()),
                bounds=[(0, 1)] * len(self.thresholds),
                method='L-BFGS-B'
            )
        
        return self.best_results