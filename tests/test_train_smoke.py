import tempfile
from pathlib import Path

from sklearn.datasets import make_classification
import joblib

from scripts.train_baseline import main as train_main


def test_train_smoke(monkeypatch, tmp_path: Path):
    # Create a tiny synthetic CSV file with the expected columns
    csv = tmp_path / "sample.csv"
    with csv.open("w", encoding="utf-8") as fh:
        fh.write("ham,hello world\n")
        fh.write("spam,win money now\n")
        fh.write("ham,how are you\n")

    # Run training with the sample CSV; pass arguments via monkeypatch of argv
    import sys

    monkeypatch.setattr(sys, 'argv', ["train_baseline", "--data", str(csv), "--model-out", str(tmp_path / 'model.joblib'), "--metrics-out", str(tmp_path / 'metrics.json'), "--cm-out", str(tmp_path / 'cm.png')])
    train_main()

    assert (tmp_path / 'model.joblib').exists()
    assert (tmp_path / 'metrics.json').exists()
