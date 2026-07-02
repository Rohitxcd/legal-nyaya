from time import time

from dotenv import load_dotenv
load_dotenv()
"""
pipeline.py — Nyaya end-to-end RAG pipeline
Orchestrates: retrieve → rerank → synthesize
Returns answer + contexts shaped for RAGAS evaluation

Usage:
    python src/pipeline.py "what is anticipatory bail?"
"""

import os
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from groq import RateLimitError, APIConnectionError, APIStatusError

# ── Import retrieval layer ────────────────────────────────────────────────────
# Run all scripts from project root: python src/pipeline.py "question"
sys.path.append(".")
from src.ingest.query import load_index, search, rerank

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(question: str, contexts: list) -> str:
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return f"""You are a legal assistant specializing in Indian Supreme Court judgements.
Answer the question based ONLY on the provided context excerpts from real judgements.
If the answer cannot be found in the context, say "I cannot find sufficient information in the provided judgements to answer this question."
Do not add any information outside of what is provided.

Context:
{context_text}

Question: {question}

Answer:"""

def call_groq_with_retry(groq_client, prompt: str, max_retries: int = 3, base_delay: float = 2.0) -> str:
    """
    Calls Groq with exponential backoff on rate limits / transient errors.
    Raises the last exception if all retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            if attempt == max_retries:
                raise
            # Groq's 429 body often includes a suggested wait time; fall back to backoff if not parseable
            delay = base_delay * (2 ** attempt)
            print(f"  Rate limited (attempt {attempt+1}/{max_retries}). Retrying in {delay:.1f}s...")
            time.sleep(delay)

        except (APIConnectionError, APIStatusError) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  Groq API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(question: str, index, metadata, model, reranker, groq_client) -> dict:
    """
    Full RAG pipeline for one question.
    Returns dict shaped for RAGAS evaluation + frontend display.
    """

    # Step 1: FAISS retrieves 20 candidates
    candidates = search(question, index, metadata, model)

    # Step 2: Cross-encoder reranks to top 5
    results = rerank(question, candidates, reranker)

    # Step 3: Extract plain text contexts (what RAGAS and the LLM both need)
    contexts = [r["text"] for r in results]

    # Step 4: Build prompt
    prompt = build_prompt(question, contexts)

 
   # Step 5: Call Groq (with retry on rate limit / transient errors)
    answer = call_groq_with_retry(groq_client, prompt)

    return {
        "question": question,
        "answer":   answer,
        "contexts": contexts,   # list of strings — RAGAS expects this format
        "sources": [            # structured metadata — frontend uses this
            {
                "title": r["title"],
                "court": r["court"],
                "year":  r["year"],
                "url":   r["url"],
                "chunk_id": r["chunk_id"],
            }
            for r in results
        ],
    }


# ── Load all models once ──────────────────────────────────────────────────────

def load_pipeline():
    """
    Load all models and index once.
    Call this at startup — reuse across multiple queries.
    """
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_NAME)

    print("Loading FAISS index...")
    index, metadata = load_index()
    print(f"Index ready. {index.ntotal} vectors.")

    print("Connecting to Groq...")
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    return index, metadata, model, reranker, groq_client


# ── Main (manual testing) ─────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print('Usage: python src/pipeline.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    index, metadata, model, reranker, groq_client = load_pipeline()
    result = run_pipeline(question, index, metadata, model, reranker, groq_client)

    print(f"\nQuestion: {result['question']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  [{s['chunk_id']}] {s['title']} ({s['year']})")
        print(f"  {s['url']}")


if __name__ == "__main__":
    main()