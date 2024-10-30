import pandas as pd
import os
import re
import gc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm

import torch

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import RobertaTokenizer, RobertaForSequenceClassification
from huggingface_hub import login

# Log in to Hugging Face with your token
token = "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"
login(token)

# Load the tokenizer and model
model_name = 'openai-community/roberta-large-openai-detector'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model3 = RobertaForSequenceClassification.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model3.eval()


# Function to classify text with the loaded Roberta model
def classify_text3(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model3(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1).tolist()[0]
    # probabilities[0] corresponds to "human-written"
    # probabilities[1] corresponds to "AI-generated"
    return predictions

# Function to get AI probability from classification
def roberta_large_openai_detector_probability(text):
    probabilities = classify_text3(text)
    ai_probability = probabilities[1]  # Probability of being AI-generated
    return ai_probability

# Load your sample DataFrame
# Assuming df_sample is already defined and has a 'text' column
dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE_with_radar_d2.csv")

df = dataset.sample(frac=1, random_state=42)

# Add a new column to store detection results from the Roberta model
df['roberta_large_openai_detector_probability'] = df['essay'].apply(roberta_large_openai_detector_probability)


df.columns

# Save the updated DataFrame to CSV
output_file_path = "C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Training_dataset\\Train_RAID_MAGE_d3.csv"
df.to_csv(output_file_path, index=False)
print(f"Saved the updated dataset with Roberta AI detection probabilities at: {output_file_path}")

