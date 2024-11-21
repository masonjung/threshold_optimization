import textlength
import formality
import sentiment
import personality
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# path of the file
path = '/Users/cynthia/Documents/MIT/'
load_file = 'roft.csv'
save_file = 'roft_features.csv'

# pre-processing the file
df = pd.read_csv(path + load_file)
df['label'] = df['model'].apply(lambda x: 0 if x in ['human', 'baseline'] else 1)
df.rename(columns={'prompt_body': 'text'}, inplace=True) #change the name of the column
df = df[['text','label']]
#df = df[0:10]

# adding features
#df['text_length'] = df['text'].apply(textlength.count_length)
df['text_length'] = df['text'].progress_apply(textlength.count_length)
df['formality'] = df['text'].progress_apply(formality.calculate_formality_score)
df[['sentiment_label', 'sentiment_score']]  = df['text'].progress_apply(lambda x: pd.Series(sentiment.classify_sentiment_transformers(x)))
col_personality = ['extroversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']
df[col_personality] = df['text'].progress_apply(lambda x: pd.Series(personality.personality_detection(x)))

# save the file
df.to_csv(path + save_file, index=False)
print(df.tail(2))