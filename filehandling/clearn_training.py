import pandas as pd

# Read the data from the file
df_train = [PATH]
df_test = [PATH]

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

df_train.columns
