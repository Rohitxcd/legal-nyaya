"""
chunker.py — splits processed judgement JSONs into overlapping chunks
Reads from data/processed/, writes to data/chunks/

Usage:
    python src/ingest/chunker.py
"""

import json
import os

# ── Config ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = "data/processed"
CHUNKS_DIR    = "data/chunks"

CHUNK_SIZE    = 512    # words per chunk
CHUNK_OVERLAP = 64     # words of overlap between chunks

# ── Core logic ────────────────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Splits a long text into overlapping word-based chunks.

    Example with chunk_size=5, overlap=2:
      words = [w1, w2, w3, w4, w5, w6, w7, w8]
      chunk_1 = w1 w2 w3 w4 w5
      chunk_2 = w4 w5 w6 w7 w8   ← starts 2 words back (overlap)
    """
    words  = text.split()
    chunks = []
    start  = 0
    step   = chunk_size - overlap   # how far to advance each time

    while start < len(words):
        end   = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += step

    return chunks


def chunk_document(doc: dict) -> list[dict]:
    """
    Takes one processed JSON doc and returns a list of chunk dicts.
    Each chunk carries metadata so we know where it came from.
    """
    chunks     = split_into_chunks(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_docs = []

    for i, chunk_text in enumerate(chunks):
        chunk_docs.append({
            "chunk_id":    f"{doc['doc_id']}_chunk_{i:03d}",
            "doc_id":      doc["doc_id"],
            "chunk_index": i,
            "total_chunks": len(chunks),
            "title":       doc["title"],
            "court":       doc["court"],
            "year":        doc["year"],
            "url":         doc["url"],
            "text":        chunk_text,
            "word_count":  len(chunk_text.split()),
        })

    return chunk_docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    processed_files = [
        f for f in os.listdir(PROCESSED_DIR) if f.endswith(".json")
    ]

    if not processed_files:
        print("No files found in data/processed/")
        return

    total_chunks = 0

    for filename in processed_files:
        input_path  = os.path.join(PROCESSED_DIR, filename)
        output_path = os.path.join(CHUNKS_DIR, filename)  # same name, different folder

        with open(input_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        chunks = chunk_document(doc)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"  {filename} -> {len(chunks)} chunks")
        total_chunks += len(chunks)

    print(f"\nDone. {len(processed_files)} documents -> {total_chunks} total chunks")
    print(f"Chunks saved to {CHUNKS_DIR}/")


if __name__ == "__main__":
    main()