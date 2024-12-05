import pandas as pd

# Read the data from the file
df_train = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv')
df_test = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_features.csv')

# check the shape
print(df_train.shape)
print(df_test.shape)


df_train["AI_written"].value_counts()


df_train["formality"].describe()


# add formality column
def add_formality_label(df):
    df["formality_label"] = df["formality"].apply(lambda x: "formal" if x > 50 else "informal")
    return df

df_train = add_formality_label(df_train)
df_test = add_formality_label(df_test)

# add length column
def add_length_column(df):
    df["length_label"] = df["text_length"].apply(lambda x: "short" if x < 1000 else ("mid" if x <= 2500 else "long"))
    return df

df_train = add_length_column(df_train)
df_test = add_length_column(df_test)


# personality
# personality_list = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
# # Create a new column 'personality' with the greatest probability
# df_train['personality'] = df_train[personality_list].idxmax(axis=1)
# df_test['personality'] = df_test[personality_list].idxmax(axis=1)

# # Display the first few rows to verify the new column
# print(df_train.head())
# print(df_test.head())

# df_test["personality"].value_counts()

# Save the new data to a new file
# df_train.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\train_features.csv', index=False)
# df_test.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\test_features.csv', index=False)



df_train.columns
