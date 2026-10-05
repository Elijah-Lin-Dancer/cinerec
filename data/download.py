"""Download and extract the MovieLens 100K dataset.

Hardened against the usual data-pipeline footguns: the URL is allow-listed and
https-only, the request uses connect/read timeouts so a stalled host cannot hang
the build, and archive members are checked so a malicious zip cannot write
outside the target directory (Zip Slip).
"""
import os
import zipfile
from urllib.parse import urlparse

import pandas as pd
import requests

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
_ALLOWED_HOSTS = {"files.grouplens.org"}


def _validated_url(url):
    """Return ``url`` when it is https on an allow-listed host, else raise."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"Refusing to download from untrusted URL: {url}")
    return url


def _safe_extract(archive, dest):
    """Extract ``archive`` into ``dest``, rejecting members that escape it."""
    dest_root = os.path.realpath(dest)
    for member in archive.infolist():
        target = os.path.realpath(os.path.join(dest_root, member.filename))
        if os.path.commonpath([dest_root, target]) != dest_root:
            raise ValueError(f"Unsafe path in archive (Zip Slip): {member.filename}")
    archive.extractall(dest)


def download_movielens_100k():
    """Download MovieLens 100K dataset and extract u.data"""
    url = _validated_url(MOVIELENS_URL)
    zip_path = os.path.join(RAW_DIR, "ml-100k.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {url}...")
        # (connect, read) timeouts — a hung server must not block the pipeline.
        r = requests.get(url, stream=True, timeout=(10, 120))
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

        # Verify file integrity
        expected_size = 4940235  # ml-100k zip is approximately 4.9MB
        actual_size = os.path.getsize(zip_path)
        if actual_size < expected_size * 0.9:
            print(f"Warning: Downloaded file size {actual_size} bytes, expected ~{expected_size} bytes")
            print("File may be corrupted. Please re-run this script.")
    if not os.path.exists(os.path.join(RAW_DIR, "ml-100k")):
        with zipfile.ZipFile(zip_path, "r") as z:
            _safe_extract(z, RAW_DIR)
        print("Extracted.")
    return pd.read_csv(
        os.path.join(RAW_DIR, "ml-100k", "u.data"), sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

if __name__ == "__main__":
    df = download_movielens_100k()
    print(f"Loaded {len(df)} ratings from {df['user_id'].nunique()} users, {df['item_id'].nunique()} movies")
    csv_path = os.path.join(RAW_DIR, "ratings.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")