import pandas as pd

# Read the data from the file
df_train = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv')
df_test = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_features.csv')

# check the shape
print(df_train.shape)
print(df_test.shape)

# personality
personality_list = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
# Create a new column 'personality' with the greatest probability
df_train['personality'] = df_train[personality_list].idxmax(axis=1)
df_test['personality'] = df_test[personality_list].idxmax(axis=1)

# Display the first few rows to verify the new column
print(df_train.head())
print(df_test.head())

df_test["personality"].value_counts()

# Save the new data to a new file
# df_train.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv', index=False)
# df_test.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_features.csv', index=False)



df_train.columns
