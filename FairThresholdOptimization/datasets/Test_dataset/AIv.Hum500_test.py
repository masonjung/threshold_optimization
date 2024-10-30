import pandas as pd 
import os

# check files on the desktop
desktop_path = 'C:/Users/minse/Desktop/AI_Human.csv/AI_Human.csv'
# files = os.listdir(desktop_path)
# print(files)

# Load the dataset
df = pd.read_csv(desktop_path)

df.columns

#count generated or not
df['generated'].value_counts()

# extgract five thousand essays
df = df.sample(5000, random_state=42)

#store df
output_path = 'C:/Users/minse/Desktop/Programming/FairThresholdOptimization/datasets/Test_dataset/AIv.Hum500_test_sampled.csv'
df.to_csv(output_path, index=False)