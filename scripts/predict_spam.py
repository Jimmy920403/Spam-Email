"""Predict spam for single text or batch CSV.

Examples:
  python scripts/predict_spam.py --text "Free money" --model models/baseline-svm-full.joblib

  python scripts/predict_spam.py --input datasets/processed/sms_spam_clean.csv --text-col text_clean --output preds.csv --model models/baseline-svm-full.joblib
"""
from pathlib import Path
import argparse
import csv
import joblib
import json
import sys
import os

# Ensure project root is importable when running the script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_model(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(p)


def predict_text(pipeline, text: str):
    proba = None
    if hasattr(pipeline, 'predict_proba'):
        proba = pipeline.predict_proba([text])[0][1]
    else:
        try:
            score = pipeline.decision_function([text])[0]
            proba = 1/(1+2.718281828459045**(-score))
        except Exception:
            proba = None
    label = 'SPAM' if (proba is not None and proba >= 0.5) else 'HAM'
    return label, proba


def predict_batch(pipeline, infile: str, text_col: str, outpath: str):
    import pandas as pd
    df = pd.read_csv(infile)
    texts = df[text_col].fillna("").tolist()
    probs = []
    if hasattr(pipeline, 'predict_proba'):
        probs = pipeline.predict_proba(texts)[:,1]
    else:
        probs = pipeline.decision_function(texts)
    df['spam_proba'] = probs
    df['label_pred'] = df['spam_proba'].apply(lambda p: 'SPAM' if p >= 0.5 else 'HAM')
    df.to_csv(outpath, index=False)
    print(f"Wrote predictions to {outpath}")


def main():
    parser = argparse.ArgumentParser(description="Predict spam for text or CSV")
    parser.add_argument('--model', required=True)
    parser.add_argument('--text', help='Single text to classify')
    parser.add_argument('--input', help='CSV input for batch prediction')
    parser.add_argument('--text-col', default='message', help='Column name for text in CSV')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV path')
    args = parser.parse_args()

    pipeline = load_model(args.model)
    if args.text:
        label, proba = predict_text(pipeline, args.text)
        print(f"Text: {args.text}\nLabel: {label}\nSpam probability: {proba}")
    elif args.input:
        predict_batch(pipeline, args.input, args.text_col, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
