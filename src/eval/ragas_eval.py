"""
ragas_eval.py — Nyaya RAGAS baseline evaluation pipeline
Loads questions → runs pipeline → scores with RAGAS → saves CSV

Metrics:
    - Faithfulness       (does answer stick to retrieved context?)
    - Answer Relevancy   (does answer address the question?)
    - Context Precision  (are retrieved chunks actually useful?)

All three are reference-free — no ground truth needed.

Usage: python src/eval/ragas_eval.py

REQUIRED VERSIONS (pin exactly these — tested working combo):
    pip install "ragas==0.1.21" "langchain-community==0.2.16" "langchain-groq==0.1.9" 
                python-dotenv pandas datasets sentence-transformers
"""

import os
import sys
import json
import time
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.append(".")
load_dotenv()

# ── Imports ───────────────────────────────────────────────────────────────────
from src.pipeline import load_pipeline, run_pipeline
from ragas.run_config import RunConfig

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_utilization
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingfaceEmbeddings  # native ragas wrapper — no langchain_huggingface needed

from langchain_groq import ChatGroq

# ── Config ────────────────────────────────────────────────────────────────────

QUESTIONS_PATH  = "data/questions.json"
OUTPUT_PATH     = "data/eval/ragas_baseline.csv"
GROQ_MODEL      = "llama-3.1-8b-instant"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
DELAY_SECONDS   = 2   # pause between pipeline calls — avoids Groq rate limits


# ── Step 1: Load questions ────────────────────────────────────────────────────

def load_questions(path: str) -> list:
    with open(path, "r") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {path}")
    return questions


# ── Step 2: Run pipeline on each question ─────────────────────────────────────

def collect_pipeline_outputs(questions: list, index, metadata, model, reranker, groq_client) -> list:
    """
    Run the full RAG pipeline for each question.
    Returns list of dicts with question, answer, contexts.
    """
    results = []

    for i, question in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {question}")

        try:
            output = run_pipeline(question, index, metadata, model, reranker, groq_client)
            results.append({
                "question": output["question"],
                "answer":   output["answer"],
                "contexts": output["contexts"],   # list of strings — RAGAS expects this
            })
            print(f"  ✓ Answer: {output['answer'][:100]}...")

        except Exception as e:
            print(f"  ✗ Pipeline failed: {e}")
            # Still append so question count stays aligned — RAGAS skips empty answers
            results.append({
                "question": question,
                "answer":   "",
                "contexts": [],
            })

        # Pause to respect Groq free tier rate limits
        if i < len(questions) - 1:
            time.sleep(DELAY_SECONDS)

    return results


# ── Step 3: Score with RAGAS ──────────────────────────────────────────────────

def run_ragas(results: list) -> pd.DataFrame:
    """
    Feed pipeline outputs into RAGAS.
    Uses Groq as evaluator LLM + ragas' native HuggingFace embeddings wrapper
    (free, no OpenAI key, no langchain_huggingface dependency needed).
    """

    # Filter out failed pipeline runs
    valid = [r for r in results if r["answer"] and r["contexts"]]
    print(f"\nScoring {len(valid)}/{len(results)} valid results with RAGAS...")

    # Build HuggingFace dataset — RAGAS expects this format
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in valid],
        "answer":   [r["answer"]   for r in valid],
        "contexts": [r["contexts"] for r in valid],   # list of lists
    })

    # Groq as evaluator LLM — free, no OpenAI key needed
    evaluator_llm = LangchainLLMWrapper(
        ChatGroq(
            model=GROQ_MODEL,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.0,
        )
    )

    # ragas' own HuggingFace embeddings wrapper for Answer Relevancy metric
    # (replaces langchain_huggingface.HuggingFaceEmbeddings — removes that
    # entire dependency chain, which was the source of the version conflicts)
    evaluator_embeddings = HuggingfaceEmbeddings(model_name=EMBED_MODEL)

    # Run evaluation
    scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_utilization],
    llm=evaluator_llm,        # keep whatever variable name you already have here
    embeddings=evaluator_embeddings,   # keep whatever variable name you already have here
    run_config=RunConfig(
        max_workers=1,   # avoid Groq rate limits
        timeout=400,
        max_retries=3,
        max_wait=90,
    ),
)

    return scores.to_pandas()


# ── Step 4: Save results ──────────────────────────────────────────────────────

def save_results(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nResults saved to {path}")


# ── Step 5: Print summary ─────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 50)
    print("RAGAS BASELINE RESULTS")
    print("=" * 50)
    print(f"Questions evaluated : {len(df)}")
    print(f"Faithfulness        : {df['faithfulness'].mean():.4f}")
    print(f"Answer Relevancy    : {df['answer_relevancy'].mean():.4f}")
    print(f"Context Utilization : {df['context_utilization'].mean():.4f}")
    print("=" * 50)

    # Flag failure cases (faithfulness < 0.7) — these go into your dashboard
    failures = df[df["faithfulness"] < 0.7]
    if len(failures) > 0:
        print(f"\nFailure cases (faithfulness < 0.7): {len(failures)}")
        for _, row in failures.iterrows():
            print(f"  - {row['question'][:80]}...")
            print(f"    Faithfulness: {row['faithfulness']:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load RAG pipeline models once
    print("Loading pipeline...")
    index, metadata, model, reranker, groq_client = load_pipeline()

    # Load questions
    questions = load_questions(QUESTIONS_PATH)

    # Run pipeline on all questions
    print("\nRunning pipeline on all questions...")
    results = collect_pipeline_outputs(questions, index, metadata, model, reranker, groq_client)

    # Score with RAGAS
    df = run_ragas(results)

    # Save + print
    save_results(df, OUTPUT_PATH)
    print_summary(df)


if __name__ == "__main__":
    main()