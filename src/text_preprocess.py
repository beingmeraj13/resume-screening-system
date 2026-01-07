import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_resume_text(text):
    """
    Cleans resume text using NLP preprocessing steps
    """
    # Lowercase
    text = text.lower()

    # Remove URLs, emails, special characters, numbers
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords & lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(clean_tokens)
