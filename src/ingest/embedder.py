"""
embedder.py — embeds all chunks and builds a FAISS index
Reads from data/chunks/, writes to data/index/

Usage:
    python src/ingest/embedder.py
"""

import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_DIR  = "data/chunks"
INDEX_DIR   = "data/index"
INDEX_PATH  = os.path.join(INDEX_DIR, "faiss.index")
META_PATH   = os.path.join(INDEX_DIR, "metadata.pkl")

MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"

# ── Load all chunks ───────────────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    all_chunks = []
    chunk_files = [f for f in os.listdir(CHUNKS_DIR) if f.endswith(".json")]

    for filename in chunk_files:
        path = os.path.join(CHUNKS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        all_chunks.extend(chunks)
        print(f"  Loaded {filename} -> {len(chunks)} chunks")

    print(f"\nTotal chunks loaded: {len(all_chunks)}")
    return all_chunks


# ── Embed ─────────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [chunk["text"] for chunk in chunks]

    print(f"\nEmbedding {len(texts)} chunks...")
    print("(This may take a minute on CPU)")

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print(f"Done. Vector shape: {vectors.shape}")  # (113, 384)
    return vectors


# ── Build FAISS index ─────────────────────────────────────────────────────────

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dimension = vectors.shape[1]        # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dimension)  # flat L2 = exact search, fine for small corpus
    index.add(vectors.astype(np.float32))
    print(f"\nFAISS index built. Total vectors: {index.ntotal}")
    return index


# ── Save ──────────────────────────────────────────────────────────────────────

def save_index(index: faiss.Index, chunks: list[dict]):
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

    # Save metadata (chunk dicts without text — just for lookup)
    metadata = []
    for chunk in chunks:
        metadata.append({
            "chunk_id":    chunk["chunk_id"],
            "doc_id":      chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "title":       chunk["title"],
            "court":       chunk["court"],
            "year":        chunk["year"],
            "url":         chunk["url"],
            "text":        chunk["text"],   # keep text so we can return it in results
        })

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to {META_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded: {MODEL_NAME}")

    chunks  = load_all_chunks()
    vectors = embed_chunks(chunks, model)
    index   = build_faiss_index(vectors)
    save_index(index, chunks)

    print("\nEmbedder done. Ready for querying.")


if __name__ == "__main__":
    main()