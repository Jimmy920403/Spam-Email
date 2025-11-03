"""Train a baseline SVM classifier for spam detection.

Writes model to `models/` and metrics to `artifacts/metrics.json` and `artifacts/metrics.txt`.
"""
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from scripts.preprocess import load_dataset, preprocess_text


def evaluate_model(pipeline, X_test, y_test):
    # pipeline handles vectorization
    y_pred = pipeline.predict(X_test)
    try:
        # prefer predict_proba
        if hasattr(pipeline, 'predict_proba'):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_score = pipeline.decision_function(X_test)
    except Exception:
        y_score = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    # ROC AUC requires positive class scores and at least two labels
    try:
        if y_score is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
        else:
            metrics["roc_auc"] = None
    except Exception:
        metrics["roc_auc"] = None

    return metrics, y_pred


def save_confusion_matrix(y_test, y_pred, out_path: Path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center')
    fig.colorbar(cax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train baseline SVM for spam detection")
    parser.add_argument("--data", default="data/sms_spam_no_header.csv")
    parser.add_argument("--model-out", default="models/baseline-svm.joblib")
    parser.add_argument("--metrics-out", default="artifacts/metrics.json")
    parser.add_argument("--cm-out", default="artifacts/confusion_matrix.png")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_dataset(args.data)
    # keep raw messages and rely on TfidfVectorizer preprocessor
    messages = df["message"].fillna("").tolist()
    y = (df["label"].str.lower() == "spam").astype(int).values

    X_train_msgs, X_test_msgs, y_train, y_test = train_test_split(messages, y, test_size=args.test_size, random_state=args.random_state)

    # Build a pipeline: TfidfVectorizer (uses preprocess_text) + linear SVM
    vect = TfidfVectorizer(max_features=5000, preprocessor=preprocess_text)
    clf = SVC(kernel='linear', probability=True, random_state=args.random_state)
    pipeline = Pipeline([
        ("vect", vect),
        ("clf", clf),
    ])

    pipeline.fit(X_train_msgs, y_train)

    metrics, y_pred = evaluate_model(pipeline, X_test_msgs, y_test)

    # Save pipeline model
    out_model = Path(args.model_out)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_model)

    # Save metrics
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))
    with open(metrics_out.with_suffix('.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(f"{k}: {v}" for k, v in metrics.items()))

    # Save confusion matrix
    save_confusion_matrix(y_test, y_pred, Path(args.cm_out))

    print("Training complete. Metrics:", metrics)


if __name__ == "__main__":
    main()
