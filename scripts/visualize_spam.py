"""Visualization utilities for spam classifier.

Generates and saves plots: class distribution, token frequencies, confusion matrix,
ROC curve, Precision-Recall curve, and threshold sweep.

Example:
  python scripts/visualize_spam.py --input data/sms_spam_no_header.csv --models-dir models --confusion-matrix --roc --pr --class-dist --token-freq --threshold-sweep
"""
from pathlib import Path
import argparse
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (confusion_matrix, roc_curve, auc, precision_recall_curve,
                             precision_score, recall_score, f1_score)

import sys
import os

# Ensure project root is on sys.path when running script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.preprocess import preprocess_text
import joblib


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(df: pd.DataFrame, out_path: Path):
    counts = df.iloc[:, 0].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_title('Class distribution')
    ax.set_xlabel('label')
    ax.set_ylabel('count')
    ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)


def top_tokens_by_class(df: pd.DataFrame, text_col: int, topn: int, out_path: Path):
    labels = df.iloc[:, 0].astype(str)
    texts = df.iloc[:, text_col].fillna('').map(preprocess_text).tolist()
    vect = CountVectorizer(ngram_range=(1, 2), min_df=2)
    X = vect.fit_transform(texts)
    feature_names = np.array(vect.get_feature_names_out())
    for label in labels.unique():
        mask = (labels == label).to_numpy()
        freqs = np.asarray(X[mask].sum(axis=0)).ravel()
        top_idx = freqs.argsort()[::-1][:topn]
        top_feats = feature_names[top_idx]
        top_vals = freqs[top_idx]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top_feats[::-1], top_vals[::-1])
        ax.set_title(f'Top {topn} tokens for {label}')
        ensure_dir(out_path.with_name(out_path.stem + f'_{label}.png'))
        fig.savefig(out_path.with_name(out_path.stem + f'_{label}.png'))
        plt.close(fig)


def plot_confusion_roc_pr(df: pd.DataFrame, text_col: int, model_path: Path, out_dir: Path):
    obj = joblib.load(model_path)
    # normalize model object to a sklearn Pipeline
    from sklearn.pipeline import Pipeline

    if isinstance(obj, dict):
        vect = obj.get('vectorizer')
        clf = obj.get('model')
        pipeline = Pipeline([('vect', vect), ('clf', clf)])
    elif isinstance(obj, Pipeline):
        pipeline = obj
    else:
        pipeline = obj
    texts = df.iloc[:, text_col].fillna('').map(preprocess_text).tolist()
    y_true = (df.iloc[:, 0].astype(str).str.lower() == 'spam').astype(int).values

    # get scores
    if hasattr(pipeline, 'predict_proba'):
        scores = pipeline.predict_proba(texts)[:, 1]
    else:
        scores = pipeline.decision_function(texts)

    y_pred = (scores >= 0.5).astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    ax.set_title('Confusion matrix (threshold=0.5)')
    fig.colorbar(cax)
    out_cm = out_dir / 'confusion_matrix.png'
    ensure_dir(out_cm)
    fig.savefig(out_cm)
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.legend()
    out_roc = out_dir / 'roc_curve.png'
    fig.savefig(out_roc)
    plt.close(fig)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision)
    ax.set_title('Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    out_pr = out_dir / 'pr_curve.png'
    fig.savefig(out_pr)
    plt.close(fig)

    # Save simple metrics
    metrics = {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)


def threshold_sweep(df: pd.DataFrame, text_col: int, model_path: Path, out_path: Path):
    obj = joblib.load(model_path)
    from sklearn.pipeline import Pipeline
    if isinstance(obj, dict):
        vect = obj.get('vectorizer')
        clf = obj.get('model')
        pipeline = Pipeline([('vect', vect), ('clf', clf)])
    elif isinstance(obj, Pipeline):
        pipeline = obj
    else:
        pipeline = obj
    texts = df.iloc[:, text_col].fillna('').map(preprocess_text).tolist()
    y_true = (df.iloc[:, 0].astype(str).str.lower() == 'spam').astype(int).values
    if hasattr(pipeline, 'predict_proba'):
        scores = pipeline.predict_proba(texts)[:, 1]
    else:
        scores = pipeline.decision_function(texts)

    thresholds = np.linspace(0.0, 1.0, 101)
    results = []
    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        results.append({
            'threshold': float(th),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        })
    df_res = pd.DataFrame(results)
    fig, ax = plt.subplots()
    ax.plot(df_res['threshold'], df_res['precision'], label='precision')
    ax.plot(df_res['threshold'], df_res['recall'], label='recall')
    ax.plot(df_res['threshold'], df_res['f1'], label='f1')
    ax.set_xlabel('threshold')
    ax.set_ylabel('score')
    ax.legend()
    ensure_dir(out_path)
    fig.savefig(out_path)
    plt.close(fig)
    df_res.to_csv(out_path.with_suffix('.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Visualize spam dataset and model outputs')
    parser.add_argument('--input', default='data/sms_spam_no_header.csv')
    parser.add_argument('--text-col', type=int, default=1, help='Index of text column (0-based)')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('--output-dir', default='reports/visualizations')
    parser.add_argument('--topn', type=int, default=20)
    parser.add_argument('--confusion-matrix', action='store_true')
    parser.add_argument('--roc', action='store_true')
    parser.add_argument('--pr', action='store_true')
    parser.add_argument('--class-dist', action='store_true')
    parser.add_argument('--token-freq', action='store_true')
    parser.add_argument('--threshold-sweep', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.input, header=None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = None
    # pick the first model found in models-dir
    mdir = Path(args.models_dir)
    if mdir.exists():
        models = list(mdir.glob('*.joblib')) + list(mdir.glob('*.pkl'))
        if models:
            model_path = models[0]

    if args.class_dist:
        plot_class_distribution(df, out_dir / 'class_distribution.png')

    if args.token_freq:
        top_tokens_by_class(df, args.text_col, args.topn, out_dir / 'top_tokens')

    if model_path and (args.confusion_matrix or args.roc or args.pr):
        plot_confusion_roc_pr(df, args.text_col, model_path, out_dir)

    if model_path and args.threshold_sweep:
        threshold_sweep(df, args.text_col, model_path, out_dir / 'threshold_sweep.png')

    print('Visualizations saved to', out_dir)


if __name__ == '__main__':
    main()
