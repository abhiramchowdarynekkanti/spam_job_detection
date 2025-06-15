# utils/preprocessing.py
from __future__ import annotations
import nltk

# Download required NLTK data packages quietly at runtime
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

# ──────────────────────────────────────────────────────────────
# 1. Ensure required NLTK assets are present
# ──────────────────────────────────────────────────────────────
def _ensure_nltk_resource(path: str, name: str | None = None) -> None:
    """
    Check if an NLTK resource exists; download silently if it doesn't.
    """
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name or path.split("/")[-1], quiet=True)

for res, path in {
    "punkt":        "tokenizers/punkt",
    "stopwords":    "corpora/stopwords",
    "wordnet":      "corpora/wordnet",
    "omw-1.4":      "corpora/omw-1.4",
}.items():
    _ensure_nltk_resource(path, res)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ──────────────────────────────────────────────────────────────
# 2. Define text columns expected in the raw DataFrame
# ──────────────────────────────────────────────────────────────
TEXT_COLS = [
    "title", "location", "department", "company_profile", "description",
    "requirements", "benefits", "employment_type", "required_experience",
    "required_education", "industry", "function",
]

# ──────────────────────────────────────────────────────────────
# 3. Text‑cleaning helper
# ──────────────────────────────────────────────────────────────
def clean_text(text: str | None) -> str:
    """Lowercase, strip URLs/digits/punctuation, remove stop‑words, lemmatize."""
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)     # remove URLs
    text = re.sub(r"\d+", " ", text)         # remove digits
    text = re.sub(r"[^a-z\s]", " ", text)    # keep letters & spaces
    text = re.sub(r"\s+", " ", text).strip() # collapse whitespace

    tokens = (
        LEMMATIZER.lemmatize(tok)
        for tok in word_tokenize(text)
        if tok not in STOPWORDS and 1 < len(tok) < 25
    )
    return " ".join(tokens)

# ──────────────────────────────────────────────────────────────
# 4. Public API
# ──────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine relevant columns into a single cleaned **text** column
    and return a DataFrame containing only that column.
    """
    df = df.copy()

    # Ensure every text column exists (empty if missing)
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    # If caller hasn’t provided a 'text' column, build one
    if "text" not in df.columns:
        df["text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    # Clean the text
    df["text"] = df["text"].apply(clean_text)

    # Return only what the vectorizer needs
    return df[["text"]]
