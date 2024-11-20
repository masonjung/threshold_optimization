# this is for the sentiment

import pandas as pd


from transformers import pipeline
from huggingface_hub import login

token = "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"
login(token, add_to_git_credential=True)


# Load a pre-trained sentiment analysis model from Hugging Face
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')


# Define a function to classify sentiment using the Hugging Face model
def classify_sentiment_transformers(text):
    # Truncate the text to fit the model's maximum token length
    if len(text) > 512:
        text = text[:512]
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']  # Return both label and confidence score

# Apply sentiment analysis to each essay in the DataFrame and split into two columns: sentiment_label and sentiment_score
df[['sentiment_label', 'sentiment_score']] = df['essay'].apply(lambda x: pd.Series(classify_sentiment_transformers(x)))

# Display the DataFrame with the new sentiment columns
print(df[['essay', 'sentiment_label', 'sentiment_score']])
