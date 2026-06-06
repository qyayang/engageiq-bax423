#!/usr/bin/env python3
"""
Rebuild embeddings.npy and embedding_ids.npy from current opportunities.csv.
Run this after any change to the CSV that modifies titles or descriptions.

Run from engageiq/ (project root):
    python3 code/rebuild_embeddings.py
Or with offline model cache:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 code/rebuild_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

_HERE    = Path(__file__).parent
DATA_DIR = _HERE.parent / "data"
CSV_PATH = DATA_DIR / "opportunities.csv"
EMB_PATH = DATA_DIR / "embeddings.npy"
IDS_PATH = DATA_DIR / "embedding_ids.npy"

MODEL_NAME  = "all-MiniLM-L6-v2"
BATCH_SIZE  = 256


def main() -> None:
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(str(CSV_PATH))
    print(f"Loaded {len(df):,} records from CSV")
    print(f"Source breakdown: {df['source'].value_counts().to_dict()}")
    print(f"url_type breakdown: {df['url_type'].value_counts().to_dict()}")

    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    ids   = df["id"].astype(str).tolist()

    print(f"\nLoading model: {MODEL_NAME}  (local_files_only=True) …")
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)

    print(f"Encoding {len(texts):,} texts in batches of {BATCH_SIZE} …")
    embs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    np.save(str(EMB_PATH), embs)
    np.save(str(IDS_PATH), np.array(ids))

    print(f"\n✅  Saved {embs.shape} embeddings → {EMB_PATH}")
    print(f"✅  Saved {len(ids)} IDs          → {IDS_PATH}")
    print(f"\n  hn_search_fallback count: "
          f"{(df['url_type'] == 'hn_search_fallback').sum()} (target: 0)")


if __name__ == "__main__":
    main()
