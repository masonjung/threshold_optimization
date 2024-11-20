import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the file
df = pd.read_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE_d3.csv')
df.shape

# columns that I neee to have
columns = ['essay', 'AI_written', 'radar_probability', 'roberta_base_openai_detector_probability', 'roberta_large_openai_detector_probability']

# Drop the columns that are not needed
df = df.drop(columns=[col for col in df.columns if col not in columns])

# save the cleaned data
df.to_csv('C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_dataset_d3_ready.csv', index=False)