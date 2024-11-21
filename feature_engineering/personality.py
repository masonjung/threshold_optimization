import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

def personality_detection(text):
    tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
    model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality")

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.squeeze()

    sigmoid = torch.nn.Sigmoid()
    probabilities = sigmoid(logits).detach().numpy()

    label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
    result = {label_names[i]: probabilities[i] for i in range(len(label_names))}

    return result

if __name__ == "__main__":
    text_input = "I prefer the quiet. The loud, busy world around me feels overwhelming. Alone, I can think, recharge, and find peace."
    print(personality_detection(text_input))

    text_input = "Life is all about energy and connection! I thrive in social spaces, meeting new people, and creating excitement."
    print(personality_detection(text_input))