# this is for the sentiment
import pandas as pd
from transformers import pipeline
from huggingface_hub import login
import torch

token = [PATH]
login(token, add_to_git_credential=True)

class SentimentAnalysis:
    def __init__(self):
        # Load a pre-trained sentiment analysis model from Hugging Face
        #self.device = torch.cuda.is_available() and 0 or -1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                            model = 'distilbert-base-uncased-finetuned-sst-2-english',
                                            device = self.device)


    def get_token_counts(self, text):
        # Tokenizer instance
        tokenizer = self.sentiment_analyzer.tokenizer
        
        # Tokenize without truncation
        tokens_full = tokenizer.encode(text, truncation=False)
        token_count_full = len(tokens_full)

        # Tokenize with truncation
        #tokens_truncated = tokenizer.encode(text, truncation=True, max_length=512) # 512 is the max_length by deafault.
        tokens_truncated = tokenizer.encode(text, truncation=True)
        token_count_truncated = len(tokens_truncated)

        return token_count_full, token_count_truncated


    # Define a function to classify sentiment using the Hugging Face model
    def classify_sentiment_transformers(self, text):
        # Truncate the text to fit the model's maximum token length
        #if len(text) > 512:
        #    text = text[:512]
        result = self.sentiment_analyzer(text, truncation=True)[0]
        return result['label'], result['score']  # Return both label and confidence score


if __name__ == "__main__":
    # Apply sentiment analysis to each essay in the DataFrame and split into two columns: sentiment_label and sentiment_score
    #df[['sentiment_label', 'sentiment_score']] = df['essay'].apply(lambda x: pd.Series(classify_sentiment_transformers(x)))

    # Display the DataFrame with the new sentiment columns
    #print(df[['essay', 'sentiment_label', 'sentiment_score']])

    text = """
    The history of cinema is a fascinating journey through technological innovation, artistic evolution, and cultural impact. It began in the late 19th century, during a period when inventors sought ways to capture and project moving images. The first motion pictures were short, silent clips, often documenting everyday activities, created using devices like the Kinetoscope, invented by Thomas Edison and William Dickson in 1891. However, the Lumière brothers in France are credited with the first public screening of motion pictures in 1895, marking the birth of cinema as a shared experience.
The Silent Era
From the late 1890s to the 1920s, the silent film era dominated. During this time, films were accompanied by live music to enhance the experience. Directors like D.W. Griffith in the United States pushed the boundaries of storytelling with landmark films such as The Birth of a Nation (1915). Meanwhile, filmmakers in Europe, such as Sergei Eisenstein in the Soviet Union, experimented with montage techniques, demonstrating how editing could manipulate time and emotion. Comedy also flourished during this period, led by iconic figures like Charlie Chaplin and Buster Keaton, whose physical humor transcended the limitations of silent cinema.
The Advent of Sound
The introduction of synchronized sound in the late 1920s revolutionized cinema. The Jazz Singer (1927), often considered the first "talkie," demonstrated the potential of sound to transform storytelling. Studios rapidly adopted sound technology, and by the 1930s, silent films had largely disappeared. This era, known as the Golden Age of Hollywood, saw the rise of major studios like MGM, Warner Bros., and Paramount, which developed the star system, making actors like Clark Gable and Greta Garbo household names. Genres like musicals, screwball comedies, and gangster films captivated audiences, while directors like Alfred Hitchcock began to explore innovative narrative techniques.
Global Influence
While Hollywood dominated, international cinema also thrived. Germany’s Weimar Republic produced groundbreaking films such as Fritz Lang’s Metropolis (1927), which explored futuristic themes. In Japan, Akira Kurosawa emerged as a master storyteller with films like Rashomon (1950), influencing filmmakers worldwide. Italian Neorealism, led by directors like Vittorio De Sica with Bicycle Thieves (1948), portrayed the struggles of ordinary people in post-war Italy. Meanwhile, India’s film industry, particularly Bollywood, became one of the largest in the world, blending music, drama, and romance in films that appealed to diverse audiences.
The Rise of New Waves
The mid-20th century brought a series of "new wave" movements. In France, the Nouvelle Vague (French New Wave) rejected traditional storytelling in favor of more experimental approaches, with directors like Jean-Luc Godard and François Truffaut leading the charge. Similarly, the British New Wave explored social realism, focusing on the working class. In the United States, the New Hollywood movement of the 1960s and 70s gave rise to auteurs like Martin Scorsese and Francis Ford Coppola, whose films (Taxi Driver, The Godfather) tackled complex themes and moral ambiguity.
The Digital Revolution
The late 20th and early 21st centuries saw cinema undergo another major transformation with the advent of digital technology. Computer-generated imagery (CGI) revolutionized filmmaking, allowing for unprecedented visual effects. Films like Jurassic Park (1993) and The Matrix (1999) showcased the potential of CGI to create immersive worlds. Digital cameras and editing tools democratized filmmaking, enabling independent creators to produce high-quality films outside traditional studio systems.
Cinema in the Streaming Era
Today, the rise of streaming platforms like Netflix, Amazon Prime, and Disney+ has reshaped how audiences consume films. The traditional theater experience faces challenges as viewers increasingly opt for on-demand content at home. However, cinema remains a powerful medium for storytelling, capable of reflecting and shaping society.
From its humble beginnings as flickering images on a screen to its current state as a global, multi-billion-dollar industry, cinema has evolved dramatically. Yet, its core purpose—capturing and sharing human stories—remains unchanged.
    """

    print(len(text))

    sentiment = SentimentAnalysis()

    # Get token counts
    token_count_full, token_count_truncated = sentiment.get_token_counts(text)
    print(f"Token count (full): {token_count_full}")
    print(f"Token count (truncated): {token_count_truncated}")

    # Get sentiment
    label, score = sentiment.classify_sentiment_transformers(text)
    print(f"Sentiment: {label}, Score: {score}")