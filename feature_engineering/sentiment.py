# this is for the sentiment
import pandas as pd
from transformers import pipeline
from huggingface_hub import login
import torch

token = "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"
login(token, add_to_git_credential=True)

class SentimentAnalysis:
    def __init__(self):
        # Load a pre-trained sentiment analysis model from Hugging Face
        #self.device = torch.cuda.is_available() and 0 or -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                            model = 'distilbert-base-uncased-finetuned-sst-2-english',
                                            device = self.device)

    # Define a function to classify sentiment using the Hugging Face model
    def classify_sentiment_transformers(self, text):
        # Truncate the text to fit the model's maximum token length
        if len(text) > 512:
            text = text[:512]
        result = self.sentiment_analyzer(text)[0]
        return result['label'], result['score']  # Return both label and confidence score


if __name__ == "__main__":
    # Apply sentiment analysis to each essay in the DataFrame and split into two columns: sentiment_label and sentiment_score
    df[['sentiment_label', 'sentiment_score']] = df['essay'].apply(lambda x: pd.Series(classify_sentiment_transformers(x)))

    # Display the DataFrame with the new sentiment columns
    print(df[['essay', 'sentiment_label', 'sentiment_score']])
