import pandas as pd

# Read the data from the file
df_train = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_features.csv')

df_train["source"].unique()

# eliminate roft
df_train_with_three = df_train[df_train["source"] != "roft"]

# check the shape
df_train_with_three.shape

#store
df_train_with_three.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_3t_features.csv', index=False)