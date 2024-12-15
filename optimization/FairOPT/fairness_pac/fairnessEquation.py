from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_f1(y_true, y_pred):
    return f1_score(y_true, y_pred)


# Interpreting equaltion
def equation_fairness(Prob_a,Prob_b):
    rule = 0.8 # threshold of 0.8 is based on 80% rule. # can be changed
    greater = max(Prob_a, Prob_b)
    smaller = min(Prob_a, Prob_b)
    evaluation = (smaller/greater) > rule # if the ratio exceeds, it is considered fair = True
    return evaluation
    # Based on Feldman, M., etc. (2015) Certifying and removing disparate impact.

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
            # Here, you can define additional logic to categorize levels of fairness
