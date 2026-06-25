"""
query.py — search the Nyaya FAISS index with a natural language question
Loads data/index/faiss.index + metadata.pkl, embeds query, returns top-k chunks

Usage:
    python src/ingest/query.py "what is anticipatory bail?"
    python src/ingest/query.py "burden of proof in criminal cases"
"""

import sys
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH = "data/index/faiss.index"
META_PATH  = "data/index/metadata.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K      = 5

# ── Load ──────────────────────────────────────────────────────────────────────

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ── Query ─────────────────────────────────────────────────────────────────────

def search(query: str, index, metadata, model, top_k=TOP_K):
    # Embed the query
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Search FAISS
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = metadata[idx]
        results.append({
            "rank":     rank + 1,
            "score":    round(float(dist), 4),   # L2 distance — lower is better
            "title":    chunk["title"],
            "court":    chunk["court"],
            "year":     chunk["year"],
            "url":      chunk["url"],
            "chunk_id": chunk["chunk_id"],
            "text":     chunk["text"],
        })

    return results


def print_results(query: str, results: list):
    print(f"\nQuery: {query}")
    print("=" * 70)

    for r in results:
        print(f"\nRank {r['rank']}  |  Score: {r['score']}  |  {r['title']}")
        print(f"Court: {r['court']}  |  Year: {r['year']}")
        print(f"URL: {r['url']}")
        print(f"Chunk: {r['chunk_id']}")
        print(f"\n{r['text'][:300]}...")
        print("-" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/ingest/query.py \"your question here\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("Loading model and index...")
    model = SentenceTransformer(MODEL_NAME)
    index, metadata = load_index()
    print(f"Index loaded. {index.ntotal} vectors ready.")

    results = search(query, index, metadata, model)
    print_results(query, results)


if __name__ == "__main__":
    main()