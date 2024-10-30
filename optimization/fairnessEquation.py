# Interpreting equaltion
def equation_fairness(Prob_a,Prob_b):
    rule = 0.8 # threshold of 0.8 is based on 80% rule.
    greater = max(Prob_a, Prob_b)
    smaller = min(Prob_a, Prob_b)
    evaluation = (smaller/greater) > rule # if the ratio exceeds, it is considered fair = True
    return evaluation
    # Based on Feldman, M., etc. (2015) Certifying and removing disparate impact.