# roberta based code

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
token = [PATH]
login(token)

# Load the tokenizer and model
model_name = 'openai-community/roberta-base-openai-detector'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model2 = RobertaForSequenceClassification.from_pretrained(model_name)

# Ensure the model is in evaluation mode
model2.eval()

def classify_text2(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model2(**inputs)
    logits = outputs.logits
    predictions = torch.softmax(logits, dim=-1).tolist()[0]
    # probabilities[0] corresponds to "human-written"
    # probabilities[1] corresponds to "AI-generated"
    return predictions

# Clasisification
def roberta_base_openai_detector_probability(text):
    probabilities = classify_text2(text)
    ai_probability = probabilities[1]  # Probability of being AI-generated
    return ai_probability


#test dataset
dataset = pd.read_csv([PATH])

dataset['roberta_base_openai_detector_probability'] = dataset['essay'].apply(roberta_base_openai_detector_probability)

dataset.columns

# Save the dataset with the new column
dataset.to_csv([PATH])
