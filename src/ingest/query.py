"""
query.py — search the Nyaya FAISS index with a natural language question
Loads data/index/faiss.index + metadata.pkl, embeds query, returns top-k chunks
Now includes cross-encoder reranking layer.

Usage:
    python src/ingest/query.py "what is anticipatory bail?"
    python src/ingest/query.py "burden of proof in criminal cases"
"""

import sys
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Config ────────────────────────────────────────────────────────────────────

INDEX_PATH      = "data/index/faiss.index"
META_PATH       = "data/index/metadata.pkl"
MODEL_NAME      = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K           = 20   # FAISS retrieves 20 candidates
RERANK_TOP_K    = 5    # cross-encoder filters down to 5

# ── Load ──────────────────────────────────────────────────────────────────────

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


# ── Query ─────────────────────────────────────────────────────────────────────

def search(query: str, index, metadata, model, top_k=TOP_K):
    query_vector = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = metadata[idx]
        results.append({
            "rank":          rank + 1,
            "faiss_score":   round(float(dist), 4),  # L2 distance — lower is better
            "title":         chunk["title"],
            "court":         chunk["court"],
            "year":          chunk["year"],
            "url":           chunk["url"],
            "chunk_id":      chunk["chunk_id"],
            "text":          chunk["text"],
        })

    return results


# ── Rerank ────────────────────────────────────────────────────────────────────

def rerank(query: str, results: list, reranker: CrossEncoder, top_k=RERANK_TOP_K):
    # Build (query, chunk_text) pairs — cross-encoder reads both together
    pairs = [[query, r["text"]] for r in results]

    # Score all pairs in one forward pass
    scores = reranker.predict(pairs)

    # Attach cross-encoder score to each result
    for result, score in zip(results, scores):
        result["ce_score"] = round(float(score), 4)

    # Sort by cross-encoder score — higher is better (unlike FAISS L2)
    reranked = sorted(results, key=lambda x: x["ce_score"], reverse=True)

    # Update ranks and return top_k
    for i, r in enumerate(reranked[:top_k]):
        r["rank"] = i + 1

    return reranked[:top_k]


# ── Print ─────────────────────────────────────────────────────────────────────

def print_results(query: str, results: list):
    print(f"\nQuery: {query}")
    print("=" * 70)

    for r in results:
        print(f"\nRank {r['rank']}  |  CE Score: {r['ce_score']}  |  FAISS: {r['faiss_score']}  |  {r['title']}")
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

    print("Loading models and index...")
    model    = SentenceTransformer(MODEL_NAME)
    reranker = CrossEncoder(RERANKER_NAME)
    index, metadata = load_index()
    print(f"Index loaded. {index.ntotal} vectors ready.")

    # Step 1: FAISS retrieves 20 candidates
    candidates = search(query, index, metadata, model)

    # Step 2: Cross-encoder reranks to top 5
    results = rerank(query, candidates, reranker)

    print_results(query, results)


if __name__ == "__main__":
    main()