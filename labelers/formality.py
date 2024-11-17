import spacy
# import pandas as pd
# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def calculate_formality_score(text):
    # Process the text with spaCy
    doc = nlp(text)

    # Initialize counters for each word class
    noun_count = 0
    adjective_count = 0
    preposition_count = 0
    article_count = 0
    pronoun_count = 0
    verb_count = 0
    adverb_count = 0
    interjection_count = 0
    total_words = 0

    # Count occurrences of each part of speech
    for token in doc:
        if token.is_alpha:  # Only consider alphabetical words
            total_words += 1
            if token.pos_ == "NOUN":
                noun_count += 1
            elif token.pos_ == "ADJ":
                adjective_count += 1
            elif token.pos_ == "ADP":  # Prepositions are tagged as "ADP" in spaCy
                preposition_count += 1
            elif token.pos_ == "DET" and token.lower_ in {"the", "a", "an"}:
                article_count += 1
            elif token.pos_ == "PRON":
                pronoun_count += 1
            elif token.pos_ == "VERB":
                verb_count += 1
            elif token.pos_ == "ADV":
                adverb_count += 1
            elif token.pos_ == "INTJ":
                interjection_count += 1

    # Calculate formal and informal scores
    formal_score = (noun_count + adjective_count + preposition_count + article_count)
    informal_score = (pronoun_count + verb_count + adverb_count + interjection_count)

    # Calculate F-score
    if total_words > 0:
        f_score = ((formal_score - informal_score) / total_words) * 100 + 50
    else:
        f_score = 50  # Neutral score if text is empty or contains no valid words

    return f_score # 50 is for formal/informal & above 60 is highly formaly



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