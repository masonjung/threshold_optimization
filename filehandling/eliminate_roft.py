import pandas as pd

# Read the data from the file
df_train = [PATH]
df_train["source"].unique()

# eliminate roft
df_train_with_three = df_train[df_train["source"] != "roft"]

# check the shape
df_train_with_three.shape

#store
df_train_with_three.to_csv([PATH])