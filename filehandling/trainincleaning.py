import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the file
df = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE.csv')

df.columns

# Rename specified columns
df.rename(columns={'POC': 'POC_probability',
                   'LGBT+': 'LGBT+_probability',
                   'WOMEN': 'WOMEN_probability',
                   'DISABLED': 'DISABLED_probability'}, inplace=True)

# Drop columns ending with _50, _60, _70, _80, _90
columns_to_drop = [col for col in df.columns if col.endswith(('_50', '_60', '_70', '_80', '_90'))]
df.drop(columns=columns_to_drop, inplace=True)

# Display the updated DataFrame
print(df.head())


# save the updated DataFrame to a new CSV file
df.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE.csv', index=False)