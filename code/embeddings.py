"""
Content embedding + FAISS ANN retrieval.
BAX-423 Lecture 5 — Embeddings & Dimensionality Reduction.

Uses Sentence-BERT (all-MiniLM-L6-v2, 384-dim) for semantic representations.
FAISS IVF index enables sub-linear approximate nearest-neighbor search at 10k+ scale.
"""
import numpy as np
import os
from pathlib import Path

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMB_DIM = 384
_HERE = Path(__file__).resolve().parent
EMB_CACHE = _HERE.parent / "data" / "embeddings.npy"
ID_CACHE = _HERE.parent / "data" / "embedding_ids.npy"


@st.cache_resource(show_spinner="Loading embedding model…")
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


@st.cache_resource(show_spinner="Building search index…")
def build_faiss_index(ids: list[str], embeddings: np.ndarray) -> tuple:
    """
    Builds a FAISS index for nearest-neighbor search.
    ≤50k vectors: IndexFlatIP (exact, ~16 MB for 10k×384, <2ms query).
    >50k vectors: IVFFlat with sufficient training data for ANN.
    Returns (index, id_array) for lookup.
    """
    n, d = embeddings.shape

    if n <= 50_000:
        # Exact inner-product search — fast and stable at this scale
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
    else:
        n_cells = max(16, min(512, int(n ** 0.5)))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = max(1, n_cells // 4)

    id_array = np.array(ids)
    return index, id_array


def retrieve_top_k(
    query_vec: np.ndarray,
    index,
    id_array: np.ndarray,
    k: int = 200,
) -> list[tuple[str, float]]:
    """Returns list of (opportunity_id, similarity_score) sorted by score desc."""
    D, I = index.search(query_vec, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx >= 0 and idx < len(id_array):
            results.append((id_array[idx], float(score)))
    return results


def load_or_compute_embeddings(df) -> tuple[np.ndarray, list[str]]:
    """Load cached embeddings or compute and cache them."""
    if EMB_CACHE.exists() and ID_CACHE.exists():
        try:
            embs = np.load(str(EMB_CACHE))
            ids = list(np.load(str(ID_CACHE), allow_pickle=True))
            if len(embs) == len(df):
                return embs, ids
        except Exception:
            pass

    texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
    ids = df["id"].tolist()

    with st.spinner(f"Computing embeddings for {len(texts):,} opportunities…"):
        embs = embed_texts(texts)

    np.save(str(EMB_CACHE), embs)
    np.save(str(ID_CACHE), np.array(ids))
    return embs, ids
