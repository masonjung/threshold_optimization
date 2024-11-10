
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
    learning_rate=0.001,
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
