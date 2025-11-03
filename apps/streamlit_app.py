import streamlit as st
from pathlib import Path
import sys
import os
import joblib
import json
import math
from io import StringIO
import io
import glob
import numpy as np
import pandas as pd
from sklearn import metrics as skl_metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Ensure project root is on sys.path so `scripts` package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess import preprocess_text


@st.cache_resource
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        obj = joblib.load(p)
        return obj
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def get_vectorizer_and_model(obj):
    # Support either a dict {'vectorizer':..., 'model':...} or a Pipeline or (vectorizer, model)
    if obj is None:
        return None, None
    if isinstance(obj, dict):
        return obj.get('vectorizer'), obj.get('model')
    # sklearn Pipeline
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(obj, Pipeline):
            # pipeline steps: vectorizer then classifier
            return None, obj
    except Exception:
        pass
    return None, obj


def predict_proba_text(vectorizer, model, text: str):
    clean = preprocess_text(text)
    # If model is a Pipeline, it will handle vectorization
    try:
        if vectorizer is None:
            # model should be pipeline
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([clean])[0][1]
            else:
                # use decision function + sigmoid
                score = model.decision_function([clean])[0]
                proba = 1 / (1 + math.exp(-score))
        else:
            X = vectorizer.transform([clean])
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0][1]
            else:
                score = model.decision_function(X)[0]
                proba = 1 / (1 + math.exp(-score))
        return float(proba)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


@st.cache_resource
def train_demo_pipeline(sample_path: str | Path):
    """Train a small demo Pipeline on the sample dataset for cloud/demo use."""
    try:
        p = Path(sample_path)
        if not p.exists():
            return None
        df = pd.read_csv(p, header=None, names=["label", "message"]).dropna()
        df["message_clean"] = df["message"].map(preprocess_text)
        y = (df["label"].str.lower() == "spam").astype(int).values
        X = df["message_clean"].tolist()
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
            ("clf", LinearSVC())
        ])
        pipe.fit(X, y)
        return pipe
    except Exception as e:
        st.error(f"Failed to train demo model: {e}")
        return None


def ensure_demo_model(default_model_path: str = "models/pipeline.joblib"):
    """Return a Pipeline model: load if exists, otherwise train on sample and best-effort save."""
    mp = Path(default_model_path)
    # Try loading existing
    if mp.exists():
        obj = load_model(str(mp))
        _, mdl = get_vectorizer_and_model(obj)
        if mdl is not None:
            return mdl
    # Train demo
    demo = train_demo_pipeline("data/sample.csv")
    if demo is not None:
        try:
            mp.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(demo, mp)
        except Exception:
            # read-only env, ignore
            pass
        return demo
    return None


def main():
    st.title("Spam Classifier — Live Demo")

    # --- Sidebar: Model management ---
    st.sidebar.header("Model & Data")
    model_source = st.sidebar.radio(
        "Model source",
        options=["Demo (auto)", "From path", "Upload .joblib"],
        index=0,
    )

    model = None
    vect = None
    model_obj = None

    if model_source == "Demo (auto)":
        model = ensure_demo_model()
        if model is None:
            st.sidebar.error("Failed to load/train demo model. Try 'From path' or upload.")
        else:
            st.sidebar.success("Demo model ready")

    elif model_source == "From path":
        model_path = st.sidebar.text_input("Model path", value="models/pipeline.joblib")
        if model_path and st.sidebar.button("Load model"):
            model_obj = load_model(model_path)
        if model_obj is None and model_path and Path(model_path).exists():
            model_obj = load_model(model_path)
        vect, model = get_vectorizer_and_model(model_obj)
        if model is None:
            st.sidebar.warning("Model not found at path.")
        else:
            st.sidebar.success("Model loaded from path")

    else:  # Upload
        uploaded = st.sidebar.file_uploader("Upload model (.joblib)", type=["joblib"])
        if uploaded is not None:
            try:
                model_obj = joblib.load(uploaded)
                vect, model = get_vectorizer_and_model(model_obj)
                if model is not None:
                    st.sidebar.success("Uploaded model loaded")
                else:
                    st.sidebar.error("Uploaded file did not contain a valid model")
            except Exception as e:
                st.sidebar.error(f"Failed to load uploaded model: {e}")

    # --- Sidebar: Metrics file (optional) ---
    st.sidebar.markdown("---")
    metrics_path = st.sidebar.text_input("Metrics path", value="artifacts/metrics.json")
    if Path(metrics_path).exists():
        try:
            with open(metrics_path, 'r', encoding='utf-8') as fh:
                metrics = json.load(fh)
            st.sidebar.subheader("Metrics")
            for k, v in metrics.items():
                st.sidebar.write(f"**{k}**: {v}")
        except Exception as e:
            st.sidebar.write(f"Failed to read metrics: {e}")

    # --- Quick predict ---
    st.header("Quick predict")

    # Example buttons
    if "example_text" not in st.session_state:
        st.session_state.example_text = ""
    col_ex1, col_ex2, _ = st.columns(3)
    with col_ex1:
        if st.button("Example SPAM"):
            st.session_state.example_text = "Free entry in 2 a wkly comp to win cash! Call now!"
    with col_ex2:
        if st.button("Example HAM"):
            st.session_state.example_text = "Hey, are we still on for lunch at 12:30?"

    user_text = st.text_area("Enter message to classify", height=120, value=st.session_state.example_text)
    if st.button("Predict"):
        if model is None:
            st.error("No model loaded. Select a model source in the sidebar.")
        else:
            proba = predict_proba_text(vect, model, user_text)
            if proba is not None:
                st.metric(label="Spam probability", value=f"{proba*100:.2f}%")
                st.write("**Label:** ", "SPAM" if proba >= 0.5 else "HAM")

    st.header("Sample data & artifacts")
    sample_csv = Path("data/sample.csv")
    if sample_csv.exists():
        st.write("Sample dataset (first 10 rows)")
        import pandas as pd

        df = pd.read_csv(sample_csv, header=None, names=["label", "message"]).head(10)
        st.table(df)
    
        # --- Interactive evaluation / visualization ---
        st.header("Interactive evaluation")
        st.write("Use the sample dataset or upload a CSV with `label,message` columns to evaluate and visualize model behavior.")
    
        uploaded = st.file_uploader("Upload CSV (label,message)", type=["csv"])
        eval_df = None
        if uploaded is not None:
            try:
                s = StringIO(uploaded.getvalue().decode('utf-8'))
                # Auto-detect header
                peek = pd.read_csv(s, nrows=1, header=None)
                s.seek(0)
                has_header = False
                if isinstance(peek.iloc[0, 0], str) and peek.iloc[0, 0].lower() in ("label", "spam", "ham"):
                    has_header = True
                eval_df = pd.read_csv(s, header=0 if has_header else None, names=["label", "message"]).dropna()
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
    
        else:
            sample_csv = Path("data/sample.csv")
            if sample_csv.exists():
                import pandas as pd
    
                eval_df = pd.read_csv(sample_csv, header=None, names=["label", "message"]).dropna()
    
        if eval_df is not None and model is not None:
            eval_df["message_clean"] = eval_df["message"].map(preprocess_text)
            y_true = (eval_df["label"].str.lower() == "spam").astype(int).values
    
            # Get scores
            def get_scores(df_messages):
                try:
                    if vect is None:
                        # Pipeline handles vectorizing
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(df_messages)[:, 1]
                        else:
                            score = model.decision_function(df_messages)
                            proba = 1 / (1 + np.exp(-score))
                    else:
                        X = vect.transform(df_messages)
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(X)[:, 1]
                        else:
                            score = model.decision_function(X)
                            proba = 1 / (1 + np.exp(-score))
                    return proba
                except Exception as e:
                    st.error(f"Failed to compute scores: {e}")
                    return None
    
            scores = get_scores(eval_df["message_clean"].tolist())
            if scores is not None:
                fpr, tpr, roc_thresh = skl_metrics.roc_curve(y_true, scores)
                precision, recall, pr_thresh = skl_metrics.precision_recall_curve(y_true, scores)
                auc = skl_metrics.auc(fpr, tpr)
    
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("ROC Curve")
                    import matplotlib.pyplot as plt
    
                    fig1, ax1 = plt.subplots()
                    ax1.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    ax1.set_xlabel('FPR')
                    ax1.set_ylabel('TPR')
                    ax1.legend()
                    st.pyplot(fig1)
    
                with col2:
                    st.subheader("Precision-Recall")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(recall, precision)
                    ax2.set_xlabel('Recall')
                    ax2.set_ylabel('Precision')
                    st.pyplot(fig2)
    
                st.subheader("Threshold & Confusion Matrix")
                thresh = st.slider("Classification threshold", min_value=0.0, max_value=1.0, value=0.5)
                y_pred = (scores >= thresh).astype(int)
                cm = skl_metrics.confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                st.write(f"Threshold: {thresh:.2f}")
                st.write("Confusion matrix:")
                st.write(cm)
                metrics = {
                    'accuracy': skl_metrics.accuracy_score(y_true, y_pred),
                    'precision': skl_metrics.precision_score(y_true, y_pred, zero_division=0),
                    'recall': skl_metrics.recall_score(y_true, y_pred, zero_division=0),
                    'f1': skl_metrics.f1_score(y_true, y_pred, zero_division=0),
                }
                st.json(metrics)

                # Token frequency explorer
                st.subheader("Token frequency (top N)")
                top_n = st.slider("Top N", 10, 50, 20)
                try:
                    from collections import Counter
                    tokens_spam = []
                    tokens_ham = []
                    for lbl, msg in zip(eval_df["label"].str.lower(), eval_df["message"].tolist()):
                        toks = preprocess_text(msg).split()
                        if lbl == "spam":
                            tokens_spam.extend(toks)
                        else:
                            tokens_ham.extend(toks)
                    cs = Counter(tokens_spam)
                    ch = Counter(tokens_ham)
                    top_spam = cs.most_common(top_n)
                    top_ham = ch.most_common(top_n)
                    colts1, colts2 = st.columns(2)
                    with colts1:
                        st.write("Top tokens — SPAM")
                        st.table(pd.DataFrame(top_spam, columns=["token", "count"]))
                    with colts2:
                        st.write("Top tokens — HAM")
                        st.table(pd.DataFrame(top_ham, columns=["token", "count"]))
                except Exception as e:
                    st.info(f"Token explorer unavailable: {e}")
    
        elif eval_df is None:
            st.info("No evaluation dataset available. Upload a CSV or include `data/sample.csv`.")

    cm_path = Path("artifacts/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix")

    # Visualizations gallery (pre-generated images if present)
    viz_dir = Path("reports/visualizations")
    if viz_dir.exists():
        st.header("Visualizations gallery")
        images = sorted(glob.glob(str(viz_dir / "*.png")))
        if images:
            for img in images:
                st.image(img)
        else:
            st.info("No images found under reports/visualizations.")

    st.markdown("---")
    if model is not None:
        # Offer model download
        try:
            buffer = io.BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            st.download_button(
                label="Download current model",
                data=buffer,
                file_name="pipeline.joblib",
                mime="application/octet-stream",
            )
        except Exception:
            pass


if __name__ == "__main__":
    main()
