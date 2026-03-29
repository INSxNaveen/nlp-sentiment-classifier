"""
generate_data.py
================
Downloads and prepares the IMDb Large Movie Review Dataset
(Maas et al., 2011) from Stanford and saves it as a clean CSV.

Usage
-----
  python generate_data.py            # download full 50 k dataset
  python generate_data.py --sample   # create a reproducible 5 k sample for quick testing

Output
------
  data/imdb_sample.csv   columns: text, label  (0=negative, 1=positive)
"""

import os
import re
import sys
import gzip
import tarfile
import shutil
import argparse
import urllib.request
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DATASET_URL  = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
RAW_DIR      = "data/raw"
ARCHIVE_PATH = os.path.join(RAW_DIR, "aclImdb_v1.tar.gz")
EXTRACT_DIR  = os.path.join(RAW_DIR, "aclImdb")
OUTPUT_CSV   = "data/imdb_sample.csv"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = downloaded / total_size * 100 if total_size > 0 else 0
    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
    print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def download_dataset():
    """Download the IMDb archive if not already present."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if os.path.exists(EXTRACT_DIR):
        print(f"  ✓ Dataset already extracted at '{EXTRACT_DIR}'")
        return
    if not os.path.exists(ARCHIVE_PATH):
        print(f"  ↓ Downloading IMDb dataset (~84 MB) from Stanford …")
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH, _progress_hook)
        print()  # newline after progress bar
        print(f"  ✓ Saved to '{ARCHIVE_PATH}'")
    print(f"  ⟳ Extracting archive …")
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    print(f"  ✓ Extracted to '{EXTRACT_DIR}'")


def clean_html(text: str) -> str:
    """Remove HTML tags inherited from some IMDb reviews."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def load_split(split: str) -> pd.DataFrame:
    """
    Parse one split (train / test) from the extracted aclImdb folder.

    Each sentiment sub-folder contains one .txt file per review.
    Label mapping: pos → 1, neg → 0
    """
    records = []
    for sentiment, label in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(EXTRACT_DIR, split, sentiment)
        for fname in os.listdir(folder):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(folder, fname)
            with open(fpath, encoding="utf-8") as f:
                text = clean_html(f.read())
            records.append({"text": text, "label": label})
    return pd.DataFrame(records)


def build_csv(sample_only: bool = False, n_sample: int = 5_000) -> None:
    """Combine train + test splits, optionally sub-sample, and write CSV."""
    os.makedirs("data", exist_ok=True)

    print("  ⟳ Loading 'train' split …")
    df_train = load_split("train")
    print("  ⟳ Loading 'test' split …")
    df_test  = load_split("test")

    df = pd.concat([df_train, df_test], ignore_index=True)
    df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)

    if sample_only:
        df = df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(n_sample // 2, random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        print(f"  ✓ Sub-sampled to {len(df):,} reviews (balanced)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved {len(df):,} reviews → '{OUTPUT_CSV}'")
    print(f"  ✓ Class balance:\n{df['label'].value_counts().to_string()}")


def build_fallback_csv(n: int = 5_000) -> None:
    """
    If the Stanford server is unreachable, generate a synthetic dataset
    from hard-coded seed phrases so the project still runs offline.
    This is clearly labelled in a 'source' column.
    """
    print("  ℹ Generating synthetic fallback dataset …")
    os.makedirs("data", exist_ok=True)

    pos_seeds = [
        "This movie was absolutely fantastic and I loved every minute of it.",
        "An incredible film with outstanding performances from the entire cast.",
        "The director did a brilliant job bringing this story to life beautifully.",
        "I was completely captivated from start to finish, truly a masterpiece.",
        "One of the best films I have ever seen, highly recommend to everyone.",
        "The cinematography is stunning and the plot keeps you on the edge.",
        "A heartwarming and deeply moving story that will stay with you forever.",
        "Exceptional screenplay and superb acting make this a must-watch film.",
        "I laughed, I cried, and I left the theatre feeling completely satisfied.",
        "This is exactly what cinema should be: powerful, emotional, and beautiful.",
    ]
    neg_seeds = [
        "This was a complete waste of time, absolutely terrible in every way.",
        "The plot made no sense and the acting was painfully bad throughout.",
        "I fell asleep halfway through because it was so incredibly boring.",
        "Do not waste your money on this disappointing and poorly made film.",
        "The worst movie I have seen this year, a total disaster from start.",
        "Dull, lifeless, and completely devoid of any creativity whatsoever.",
        "The script was awful and the characters were entirely unconvincing.",
        "A tedious, predictable mess that fails on every conceivable level.",
        "I wanted to walk out after the first ten minutes, it was that bad.",
        "This film has no redeeming qualities and is best avoided at all costs.",
    ]

    rng     = np.random.default_rng(RANDOM_STATE)
    half    = n // 2
    texts   = []
    labels  = []

    for seeds, label in [(pos_seeds, 1), (neg_seeds, 0)]:
        for i in range(half):
            base  = seeds[i % len(seeds)]
            # Cheap augmentation: shuffle word order slightly
            words  = base.split()
            if len(words) > 6:
                cut   = rng.integers(2, len(words) - 2)
                words = words[cut:] + words[:cut]
            texts.append(" ".join(words))
            labels.append(label)

    df = pd.DataFrame({"text": texts, "label": labels, "source": "synthetic"})
    df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Synthetic dataset saved ({len(df):,} rows) → '{OUTPUT_CSV}'")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare IMDb dataset for the sentiment classifier.")
    parser.add_argument("--sample", action="store_true",
                        help="Save only 5 000 reviews instead of the full 50 000.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate a small synthetic dataset (no internet required).")
    args = parser.parse_args()

    print("\n═══════════════════════════════════════════════════════════")
    print("  IMDb Sentiment Dataset Preparation")
    print("═══════════════════════════════════════════════════════════\n")

    if args.synthetic:
        build_fallback_csv(n=5_000)
    else:
        try:
            download_dataset()
            build_csv(sample_only=args.sample)
        except Exception as exc:
            print(f"\n  ✗ Download failed: {exc}")
            print("  → Falling back to synthetic dataset …\n")
            build_fallback_csv(n=5_000)

    print("\n  Done. Run:  python classifier.py --train\n")


if __name__ == "__main__":
    main()
