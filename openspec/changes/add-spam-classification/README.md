# Spam classification — Phase 1 (Baseline)

This directory contains the OpenSpec proposal and tasks for the spam classification baseline.

Quick start (local):

1. Create a Python virtualenv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Download the dataset (will save to `data/sms_spam_no_header.csv`):

```powershell
python scripts/download_dataset.py
```

3. Run training (creates `models/baseline-svm.joblib` and `artifacts/metrics.json`):

```powershell
python scripts/train_baseline.py
```

4. Run tests:

```powershell
pytest -q
```

Notes:
- The dataset is not committed — use the download script or provide your own CSV with `label,message` rows.
- If you want to use logistic regression instead of SVM, update `scripts/train_baseline.py` or open a follow-up proposal.
