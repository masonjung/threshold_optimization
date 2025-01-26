#import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
#from tqdm import tqdm

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
        
        max_iterations=10**4,      # Maximum number of iterations.
        learning_rate = 1e-3,      # Adjusted learning rate. For threshold adjustment.
        tolerance=1e-4,            # Convergence criterion (minimum change in thresholds between iterations).
        penalty=10,                # Penalty applied if the performance metrics score (acc and f1) is below the minimum threshold.
        
        path = None,               # For the log: Path to save the results.
        group_column = None        # For the log: Column name of the group.
    ): 
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        
        self.group_indices = group_indices
        self.delta_performance = 0.2 #0.3 for 0.21 #0.2 until disparity = 0.25
        #self.min_acc = min_acc
        self.min_acc = {key: value - self.delta_performance for key, value in min_acc.items()}
        #self.min_f1 = min_f1
        self.min_f1 = {key: value - self.delta_performance for key, value in min_f1.items()}
        self.min_disparity = min_disparity
        
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.penalty = penalty
        
        self.path = path
        self.group_column = group_column
        
        self.thresholds = {key: 1.0 for key in self.group_indices}
        self.group_losses = {key: 0.0 for key in self.group_indices}
        self.is_convergence = False


        self.min_ppr_log = 1.0
        self.min_ppr_dic = []
        #self.summary_path   = self.path + f"//convergence_per_group_disparity.txt"
        #if os.path.exists(self.summary_path):
        #    os.remove(self.summary_path)
        
        #for disparity in self.min_disparities:
        #    file_path = self.path + f"/thresholds_{group_column}_disparity_{str(disparity).replace('.', '_').ljust(4, '0')}.txt"
        #    if os.path.exists(file_path):
        #        os.remove(file_path)
        #    #print(file_path)
        
        
        #with open(self.summary_path, "a") as f:
        #    f.write(f'{"=" * 70}\n')
        #    f.write(f"Min ACC threshold: {self.min_acc}\n")
        #    f.write(f"Min F1 threshold: {self.min_f1}\n")
        #    f.write(f"Acceptable disparities: {self.min_disparities}\n")
        #    f.write(f"Max iterations: {self.max_iterations}\n")
        #    f.write(f"Learning rate: {self.learning_rate}\n")
        #    f.write(f"Tolerance: {self.tolerance}\n")
        #    f.write(f"Penalty: {self.penalty}\n\n")
            
        #    f.write(f'{"=" * 70}\nGroup: {self.group_column} - {self.group_indices.keys()}\n')
    

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
        #print(f'1. self.thresholds: {thresholds}')       
        if not hasattr(self, 'first_minimize'):
            self.thresholds = {key: thresholds[i] for i, key in enumerate(self.group_indices.keys())}
            self.first_minimize = True
        
        #self.thresholds = thresholds  # Update the current thresholds
        #self.thresholds = {key: thresholds[i] for i, key in enumerate(self.group_indices.keys())}
        #print(f'2. self.thresholds: {self.thresholds}')
        print(self.thresholds)
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
            group_loss = 1 - acc  # Minimize loss based on accuracy
            #if acc < self.min_acc[group]:
            #    group_loss += self.penalty * (self.min_acc[group] - acc) ** 2  # L2 penalty
                #print("INSIDE acc < self.min_acc[group]")
            #if f1 < self.min_f1[group]:
            #    group_loss += self.penalty * (self.min_f1[group] - f1) ** 2  # L2 penalty
                #print("INSIDE f1 < self.min_f1[group]")
            if ppr_disparity > self.min_disparity:
                group_loss += self.penalty * (self.min_disparity - ppr_disparity) ** 2  # L2 penalty
                #print("INSIDE ppr_disparity > self.min_disparity")
                
            total_loss += group_loss  # Accumulate the total loss
            if group_loss <= self.group_losses[group]:
                factor = 1
            else: 
                factor = -0.5 #0.5
            #factor = 1
            #print(f'{group} \t\t ',
            #      f'factor: {factor} - {group_loss <= self.group_losses[group]} ',
            #      f'- group_loss: {group_loss} - self.group_losses[group]: {self.group_losses[group]}')
            self.group_losses[group] = group_loss
            
            #self.thresholds[group] -= self.learning_rate * (group_loss * (1 if group_loss >= 0 else -1))  # Aumentar o disminuir según la pérdida
            self.thresholds[group] -= self.learning_rate * group_loss * factor  # Update the threshold
            self.thresholds[group] = np.clip(self.thresholds[group], 0.00005, 0.99995)  # Ensure thresholds stay within bounds
            #print (f'group_loss: {group_loss}')
                           
        #acc_check = all( (acc_dict[group] >= self.min_acc[group]) for group in acc_dict)
        #f1_check  = all( (f1_dict[group]  >= self.min_f1[group] ) for group in f1_dict )
        acc_check = all( (acc_dict[group] > 0 ) for group in acc_dict)
        f1_check  = all( (f1_dict[group]  > 0 ) for group in f1_dict )
        acc = accuracy_score(self.y_true, final_y_pred)
        f1 = f1_score(self.y_true, final_y_pred, zero_division=1)
        
        
        
        confusion_matrix_df = self.update_confusion_matrix(
            group, group_y_true, group_y_pred, confusion_matrix_df
        )
        #print(f'confusion_matrix_df: \n{confusion_matrix_df}')
        ppr_values = confusion_matrix_df['PPR'].fillna(0).values
        ppr_disparity = ppr_values.max() - ppr_values.min()
        ppr_check  = ( ppr_disparity  <= self.min_disparity )
        
        if ppr_disparity > 0 and ppr_disparity < self.min_ppr_log:
            self.min_ppr_log = ppr_disparity
            self.min_ppr_dic.append(ppr_disparity)
            print(f'ppr_disparity: {ppr_disparity} - self.min_ppr_log: {self.min_ppr_log}')
        
        #if acc_check and f1_check and ppr_check == True:
        print("="*100)
        print(f'acc_check: {acc_check} - self.min_acc[group]: {self.min_acc[group]} - acc: {acc}')
        print(f'acc_dict: {acc_dict}\n')
        print(f'f1_check: {f1_check} - self.min_f1[group]: {self.min_f1[group]} - f1: {f1}')
        print(f'f1_dict: {f1_dict}\n')
        print(f'ppr_check: {ppr_check} - self.min_disparity: {self.min_disparity} - ppr_disparity: {ppr_disparity}')
        print(f'self.thresholds: {self.thresholds}')
        print(f'self.min_ppr_log: {self.min_ppr_log}')
        print(f'self.min_ppr_dic: {self.min_ppr_dic}')
        #self.is_convergence = True
        print (f"total_loss: {total_loss}")
        print("="*100)
        #print(f'3. self.thresholds: {self.thresholds}')
        if acc_check and f1_check and ppr_check == True:
            self.is_convergence = True
            print("Convergence reached")
            print("|"*100)
            #stop because the convergence is reached
            raise ConvergenceReached  # Raise the exception to stop optimization
               
        return total_loss


    
    def optimize(self):
        print([(0, 1)] * len(self.thresholds))
        #list_thresholds = [0.01 ]* len(self.thresholds)
        print(f'optimize - self.thresholds: {self.thresholds}')

        # Call the optimizer
        try:
            result = minimize(
                fun     = self.loss_function, 
                x0      = list(self.thresholds.values()), # list_thresholds,  # Initial guess for thresholds
                bounds  = [(0, 1)] * len(self.thresholds),  # Ensure thresholds stay within bounds
                method  = 'L-BFGS-B'  # You can choose other methods if needed
                #options={'maxiter': 20}  # Control max calls to "fun"
            )
            # Update the optimized thresholds
            #self.thresholds = result.x  # Set the optimized thresholds
            print("Optimized thresholds:", self.thresholds)
            return self.is_convergence
        except ConvergenceReached:
            print("Optimization stopped early due to convergence.")
            return self.is_convergence