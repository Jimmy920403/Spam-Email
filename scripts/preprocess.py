"""Preprocessing utilities for spam classification.

Provides:
- load_dataset(path)
- preprocess_text(text)
- vectorize(corpus, max_features=5000)
"""
from typing import Tuple, List
from pathlib import Path
import re
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    # The Packt CSV has no header: label,message
    df = pd.read_csv(p, header=None, names=["label", "message"], encoding="utf-8")
    return df


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove non-alphanumeric (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def vectorize(corpus: List[str], max_features: int = 5000) -> Tuple[TfidfVectorizer, object]:
    vect = TfidfVectorizer(max_features=max_features)
    X = vect.fit_transform(corpus)
    return vect, X


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick preprocess run")
    parser.add_argument("--in", dest="infile", default="data/sms_spam_no_header.csv")
    args = parser.parse_args()
    df = load_dataset(args.infile)
    df["message_clean"] = df["message"].fillna("").map(preprocess_text)
    print(df.head(3).to_dict(orient="records"))
