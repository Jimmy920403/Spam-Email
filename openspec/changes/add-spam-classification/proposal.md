## Why
Spam and phishing remain a high-impact threat in many environments. Creating a clear, testable machine-learning capability for spam email classification will provide a foundation for further improvements (models, retraining, deployment). This change introduces the project capability and an initial baseline (phase1) to validate data ingestion, training, evaluation, and CI checks.

**Assumption:** You mentioned both logistic regression and SVM; for Phase 1 the plan specifies an SVM baseline. I'll follow that plan for the baseline. Logistic regression can be added as an alternate model in a follow-up phase.

## What Changes
- Add an OpenSpec capability for spam email classification under `openspec/specs/ml/`.
- Add a change proposal that defines Phase 1 (baseline) using a Support Vector Machine (SVM) model trained on the Packt dataset.
- Provide `tasks.md` with a concrete implementation checklist for Phase 1 and empty placeholders for later phases (phase2, phase3...).

## Data Source
- Primary dataset: `https://github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/blob/master/Chapter03/datasets/sms_spam_no_header.csv`
  - Use the raw file URL in scripts (convert GitHub UI link to raw: `https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/master/Chapter03/datasets/sms_spam_no_header.csv`).

## Phase 1 — Baseline
Goal: Build a working, reproducible baseline spam classifier using classical ML (SVM). Deliverables:
- Data ingestion and preprocessing script (tokenization, lowercasing, stop-word removal optional, TF-IDF vectorization).
- Train/evaluate pipeline using scikit-learn (SVM) with train/test split and cross-validation.
- Evaluation metrics: accuracy, precision, recall, F1, ROC AUC; include confusion matrix.
- A small reproducible notebook and a CLI script `scripts/train_baseline.py` to run training and produce an artifact (`models/baseline-svm.joblib`).
- Unit tests for preprocessing and a smoke test that training completes and achieves non-trivial performance (e.g., F1 > 0.80 on the provided dataset — threshold configurable).
- CI job to run the training/test smoke checks and report results.

## Later phases (placeholders)
- Phase 2: [empty — future enhancements such as feature engineering, hyperparameter tuning, or logistic regression comparison]
- Phase 3: [empty — potential deployment, model monitoring, incremental retraining]

## Impact
- Affected files: new scripts under `scripts/`, models under `models/`, and CI workflows under `.github/workflows/`.
- No breaking changes to existing code.

## Rollout Plan
1. Add this proposal and the spec delta (this change).
2. Implement Phase 1 in a follow-up PR referencing this change-id (add-spam-classification).
3. Add CI job to run the smoke tests for the baseline.

## Acceptance Criteria
- `scripts/train_baseline.py` ingests the dataset and produces a saved model artifact.
- `openspec validate add-spam-classification --strict` passes for the provided spec delta.
- CI runs the smoke tests and returns green for the training smoke check.

## Notes
- Dataset licensing and redistribution: ensure the dataset license allows use (Packt sample datasets are typically for learning; confirm before redistribution). If redistribution is disallowed, include a download script that fetches the raw file at runtime instead of committing the data.
