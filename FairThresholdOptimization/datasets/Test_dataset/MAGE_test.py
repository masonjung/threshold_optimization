import pandas as pd
from datasets import load_dataset


# Load the MAGE dataset from Hugging Face
dataset = load_dataset("yaful/MAGE", split = "test")

# Convert the 'test' split to a Pandas DataFrame
df_test = pd.DataFrame(dataset)

# Display the first few rows of the test data
print(df_test.head())

# Take 2000 of the dataset using the sample() method
mage_test = df_test.sample(n=2000, random_state=42)

# check
mage_test.head()
mage_test.shape

# save the file in the datasets folder
mage_test.to_csv('datasets/Test_dataset/MAGE_test.csv', index=False)








