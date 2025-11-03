import streamlit as st
from pathlib import Path
import joblib
import json
import math
from io import StringIO
import numpy as np
from sklearn import metrics as skl_metrics

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


def main():
    st.title("Spam Classifier — Live Demo")

    st.sidebar.header("Model & Data")
    model_path = st.sidebar.text_input("Model path", value="models/baseline-svm.joblib")
    if not model_path:
        st.sidebar.warning("Specify model path")

    load_button = st.sidebar.button("Load model")
    model_obj = load_model(model_path) if model_path else None
    vect, model = get_vectorizer_and_model(model_obj)

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

    st.header("Quick predict")
    user_text = st.text_area("Enter message to classify", height=120)
    if st.button("Predict"):
        if not model:
            st.error("No model loaded. Check the model path or press Load model.")
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
                import pandas as pd
    
                eval_df = pd.read_csv(s, header=None, names=["label", "message"]).dropna()
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
    
        elif eval_df is None:
            st.info("No evaluation dataset available. Upload a CSV or include `data/sample.csv`.")

    cm_path = Path("artifacts/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion matrix")

    st.markdown("---")
    st.write("Model file:", model_path)


if __name__ == "__main__":
    main()
