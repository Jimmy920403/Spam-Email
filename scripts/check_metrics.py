"""Check metrics produced by training and exit non-zero if below thresholds.

Usage:
    python scripts/check_metrics.py --metrics artifacts/metrics.json --min-f1 0.8
"""
from pathlib import Path
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Check metrics JSON against thresholds")
    parser.add_argument("--metrics", default="artifacts/metrics.json")
    parser.add_argument("--min-f1", type=float, default=0.8)
    args = parser.parse_args()

    p = Path(args.metrics)
    if not p.exists():
        print(f"Metrics file not found: {p}")
        sys.exit(2)

    data = json.loads(p.read_text(encoding='utf-8'))
    f1 = data.get("f1")
    if f1 is None:
        print("No 'f1' metric found in metrics file")
        sys.exit(2)

    print(f"F1: {f1}; required >= {args.min_f1}")
    if float(f1) < args.min_f1:
        print("F1 below threshold — failing")
        sys.exit(1)

    print("Metrics OK")


if __name__ == "__main__":
    main()
