import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Function to calculate formality score using NLTK

def calculate_formality_score(text):
    # Tokenize text and initialize counters for each word class
    words = word_tokenize(text)
    total_words = len(words)

    # Initialize counters for each word class
    noun_count = 0
    adjective_count = 0
    preposition_count = 0
    article_count = 0
    pronoun_count = 0
    verb_count = 0
    adverb_count = 0
    interjection_count = 0

    # POS tagging using NLTK
    pos_tags = pos_tag(words)

    # Count occurrences of each part of speech using POS tags
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            noun_count += 1
        elif tag.startswith('JJ'):
            adjective_count += 1
        elif tag == 'IN':  # Prepositions
            preposition_count += 1
        elif word.lower() in {"the", "a", "an"}:
            article_count += 1
        elif tag.startswith('PRP'):
            pronoun_count += 1
        elif tag.startswith('VB'):
            verb_count += 1
        elif tag.startswith('RB'):
            adverb_count += 1
        elif tag == 'UH':
            interjection_count += 1

    # Calculate formal and informal scores
    formal_score = (noun_count + adjective_count + preposition_count + article_count)
    informal_score = (pronoun_count + verb_count + adverb_count + interjection_count)

    # Calculate F-score
    if total_words > 0:
        f_score = ((formal_score - informal_score) / total_words) * 100 + 50
    else:
        f_score = 50  # Neutral score if text is empty or contains no valid words

    return f_score  # 50 is for formal/informal & above 60 is highly formal


if __name__ == "__main__":
    if __name__ == "__main__":
        # Create a DataFrame with 20 example essays
        essays = [
            "The quick brown fox jumps over the lazy dog.",
            "This essay will discuss the significance of sustainable development.",
            "John went to the store and bought some apples.",
            "The project was completed on time, with all stakeholders involved.",
            "In this modern era, technological advancements shape our society.",
            "The cat sat on the mat, and everyone was happy.",
            "Global warming is one of the most pressing issues of our time.",
            "The experiment concluded with fascinating results.",
            "He is a great friend, always there when you need him.",
            "Artificial intelligence can revolutionize various industries.",
            "As the sun set, they walked along the beach.",
            "The government has introduced new policies to combat climate change.",
            "Education plays a vital role in personal and professional development.",
            "After a long day, she enjoyed a peaceful evening at home.",
            "The conference highlighted key challenges in cybersecurity.",
            "She was thrilled to receive the award for her hard work.",
            "The economic implications of the policy were profound.",
            "He quickly realized the importance of teamwork.",
            "This analysis aims to uncover trends in social behavior.",
            "The best way to achieve success is through hard work and dedication."
        ]

        df = pd.DataFrame(essays, columns=["Essay"])

        # Apply the formality score calculation to each essay
        df["F-score"] = df["Essay"].apply(calculate_formality_score)

        # Display the DataFrame
        print(df)
