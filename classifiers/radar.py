import transformers
import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
# device = "cuda:0" # example: cuda:0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector_path_or_id = "TrustSafeAI/RADAR-Vicuna-7B"
detector = transformers.AutoModelForSequenceClassification.from_pretrained(detector_path_or_id)
tokenizer = transformers.AutoTokenizer.from_pretrained(detector_path_or_id)
detector.eval()
detector.to(device)

torch.cuda.is_available()


# Load dataset

dataset = pd.read_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Test_dataset\\test_rl.csv")
df = dataset.sample(frac=1, random_state=42)  # Sample 10% of the dataset
df.shape
df.columns


# Verify that the 'essay' column exists
if 'essay' not in df.columns:
    raise KeyError("The column 'essay' does not exist in the dataset.")
else:
    print("Column 'essay' found.")


# Define a function to detect AI-generated probability for a given text
def get_ai_generated_probability(text_input):
    with torch.no_grad():
        inputs = tokenizer(text_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_probs = F.log_softmax(detector(**inputs).logits, -1)[:, 0].exp().tolist()
    return output_probs[0]  # Return the first probability

# Add a new column called 'radar_probability' to store detection results
df['radar_probability'] = df['essay'].apply(get_ai_generated_probability)


df.columns

# Save the updated dataset to CSV
df.to_csv("C:\\Users\\minse\\Desktop\\Programming\\FairThresholdOptimization\\datasets\\Test_dataset\\test_ra_rl.csv", index=False)




