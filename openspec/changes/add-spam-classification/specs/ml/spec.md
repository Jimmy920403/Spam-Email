## ADDED Requirements

### Requirement: Spam Email Classification — Baseline
The project SHALL provide a reproducible baseline machine-learning pipeline to classify SMS/email messages as spam or ham (not spam). The baseline SHALL use classical machine learning (SVM) and include data ingestion, preprocessing, training, evaluation, and a saved model artifact.

#### Scenario: Successful baseline training
- **WHEN** the contributor runs the baseline training script with the provided dataset
- **THEN** the pipeline completes without errors and produces a saved model artifact under `models/` and outputs evaluation metrics (accuracy, precision, recall, F1, ROC AUC)

#### Scenario: Data ingestion
- **WHEN** the download script is executed
- **THEN** the dataset CSV is fetched from the GitHub raw URL and saved to `data/sms_spam_no_header.csv` (unless local dataset is already present)

### Requirement: Baseline evaluation
The baseline pipeline SHALL compute and report standard classification metrics: accuracy, precision, recall, F1, and ROC AUC, and produce a confusion matrix image.

#### Scenario: Metric reporting
- **WHEN** training completes
- **THEN** a machine-readable metrics file (`artifacts/metrics.json`) is written containing the metrics and a human-readable `artifacts/metrics.txt` is created for convenience

### Requirement: CI smoke checks
CI SHALL run a smoke test that executes the preprocessing and training on a small subset and fails the job if training does not complete or if performance is below a configurable threshold.

#### Scenario: CI validation failure
- **WHEN** the CI smoke test runs and the F1 score is below the configured threshold
- **THEN** the CI job fails and outputs the metrics for debugging
