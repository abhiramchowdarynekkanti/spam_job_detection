# utils/preprocessing.py
from __future__ import annotations

import re
import string
from pathlib import Path
from typing import List

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ──────────────────────────────────────────────────────────────
# 1️⃣  Ensure required NLTK assets are present
# ──────────────────────────────────────────────────────────────
_NLTK_PACKAGES: List[str] = ["punkt", "stopwords", "wordnet", "omw-1.4"]

for pkg in _NLTK_PACKAGES:
    try:
        nltk.data.find(f"corpora/{pkg}")  # punkt lives in tokenizers/, but this works for all
    except LookupError:
        nltk.download(pkg, quiet=True)
import nltk

# Always ensure the required packages are downloaded
def ensure_nltk_resource(resource: str, download_if_missing=True):
    try:
        nltk.data.find(resource)
    except LookupError:
        if download_if_missing:
            nltk.download(resource.split("/")[-1], quiet=True)

# Ensure all required resources
ensure_nltk_resource("tokenizers/punkt")
ensure_nltk_resource("corpora/stopwords")
ensure_nltk_resource("corpora/wordnet")
ensure_nltk_resource("corpora/omw-1.4")

# Initialize once
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ──────────────────────────────────────────────────────────────
# 2️⃣  Columns we care about (if present)
# ──────────────────────────────────────────────────────────────
TEXT_COLS: List[str] = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

# ──────────────────────────────────────────────────────────────
# 3️⃣  Text‑cleaning helper
# ──────────────────────────────────────────────────────────────
def _clean_text(text: str | None) -> str:
    """Lower‑case, strip URLs / digits / punctuation, remove stop‑words, lemmatize."""
    if not isinstance(text, str):
        text = ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"\d+", " ", text)               # remove digits
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters & spaces
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace

    tokens = (
        LEMMATIZER.lemmatize(tok)
        for tok in word_tokenize(text)
        if tok not in STOPWORDS and 1 < len(tok) < 25
    )
    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────
# 4️⃣  Public API
# ──────────────────────────────────────────────────────────────
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of *df* with a single column **text** that is
    cleaned and ready for TF‑IDF or other NLP pipelines.
    """
    df = df.copy()

    # Ensure every expected text column exists as a string
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str).fillna("")

    # Build unified raw text (existing 'text' column wins if supplied)
    if "text" not in df.columns:
        df["text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    # Clean it
    df["text"] = df["text"].apply(_clean_text)

    # Return only the column your model needs
    return df[["text"]]

