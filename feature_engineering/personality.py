import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

class PersonalityTraits:
    def __init__(self):
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("Minej/bert-base-personality")
        self.model = BertForSequenceClassification.from_pretrained("Minej/bert-base-personality").to(self.device) 


    def personality_detection(self, text):
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.squeeze()

        sigmoid = torch.nn.Sigmoid()
        probabilities = sigmoid(logits).detach().numpy()

        label_names = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']
        result = {label_names[i]: probabilities[i] for i in range(len(label_names))}

        return result

if __name__ == "__main__":
    detector = PersonalityTraits()

    text_input = "I prefer the quiet. The loud, busy world around me feels overwhelming. Alone, I can think, recharge, and find peace."
    print(detector.personality_detection(text_input))

    text_input = "Life is all about energy and connection! I thrive in social spaces, meeting new people, and creating excitement."
    print(detector.personality_detection(text_input))