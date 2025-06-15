import re, string, pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk

# Ensure required NLTK resources are downloaded
for pkg in ("stopwords", "punkt", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

TEXT_COLS = [
    "title", "location", "department", "company_profile", "description",
    "requirements", "benefits", "employment_type", "required_experience",
    "required_education", "industry", "function",
]

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[0-9]", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in word_tokenize(text)
        if tok not in STOPWORDS and len(tok) < 25
    ]
    return " ".join(tokens)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Fill NA and build unified 'text' column
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("")
    if "text" not in df.columns:
        df["text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    # Clean the text
    df["text"] = df["text"].apply(clean_text)

    # Only return text column (used for TF-IDF)
    return df[["text"]]

