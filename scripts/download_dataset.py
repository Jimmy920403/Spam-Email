"""Download dataset script for spam classification.

Usage:
    python scripts/download_dataset.py --url <raw_csv_url> --out data/sms_spam_no_header.csv

By default the script downloads the Packt sample dataset raw URL.
"""
from pathlib import Path
import argparse
import requests


DEFAULT_URL = (
    "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/"
    "master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download spam dataset CSV")
    parser.add_argument("--url", default=DEFAULT_URL, help="Raw URL to CSV file")
    parser.add_argument("--out", default="data/sms_spam_no_header.csv", help="Output path")
    args = parser.parse_args()

    out_path = Path(args.out)
    print(f"Downloading {args.url} → {out_path}")
    download(args.url, out_path)
    print("Done")


if __name__ == "__main__":
    main()
