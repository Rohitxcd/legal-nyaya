# Nyāya Project Journal

Append-only. One entry per work session: what I did, result, what I learned/decided, next.
Raw outputs live in `data/eval/`, not here.

---

## Project summary (written 2026-10-17, covers everything before this date)

**Goal:** a semantic search engine over Indian Supreme Court judgements, built as the main ML portfolio piece for MAANG-level internship applications (target window: Jan-Feb 2027).

**Architecture:** fine-tuned sentence-transformer -> FAISS retrieval -> cross-encoder reranking -> Groq LLM synthesis. RAGAS measures before/after fine-tuning. Planned: FastAPI on Railway, Next.js on Vercel, Supabase query logging, HuggingFace model, private /dashboard page for engineering metrics and annotated failure cases.

**Locked decisions**
- No LangChain in the core pipeline (every layer must be explainable from first principles). LangChain only as a thin adapter for the RAGAS evaluator.
- all-MiniLM-L6-v2 is the intentionally weak pre-fine-tune baseline.
- The 50-question golden set is permanent: a regression suite for every future pipeline change.
- Reference-free RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision) are the backbone. "Insufficient information" answers are correct behaviour, not failures.
- Dependencies pinned (ragas==0.1.21, langchain-community==0.2.16, langchain-groq==0.1.9). Never `pip install -U`; install pinned packages in one command.
- Sanity-check a single question before any full 50-question run.
- Hybrid search (BM25 + RRF) and GraphRAG deferred, not abandoned.

**History**
- *Week 2 (Sep 2026):* core pipeline working end to end on 4 judgements / 117 chunks. Two-stage retrieval (FAISS top-20 -> cross-encoder -> top-5). Synthesis and RAGAS judge both moved to openai/gpt-oss-120b on Groq (old llama-3.1-8b-instant deprecated, Ollama judge removed). Fixed two bugs: deprecated model name, and `from time import time` shadowing the `time` module (broke retry sleep).
- *Oct 2026:* scaled the corpus from 4 judgements to the AWS Open Data Supreme Court set. 2023 English done: 854 judgements, 63,928 chunks (160 words / 30 overlap), index rebuilt with MiniLM. doc_id comes from the neutral citation (e.g. 2023INSC600). Plan: baseline + fine-tune on 2023, then scale to more years (7k+ target) with the fine-tuned model.

---

## 2026-10-17: Retrieval baseline

- Wrote 50 golden questions (data/golden_questions.json, answer key in `doc_ids`). Built retrieval-only eval: src/eval/retrieval_eval.py.
- Result (prod config: FAISS top-20 -> rerank -> top-5, 2023 corpus): Hit 88%, Recall 80%, MRR 82.2%, NDCG 82.2%. FAISS alone at K=5: Hit 82%, NDCG 74.5%. Saved in data/eval/retrieval_baseline_2026-10-17_1140.json.
- Reranker adds ~8 NDCG points over FAISS alone.
- Inspected the 6 misses (q027, q030, q032, q037, q038, q044): 4 are false misses (search returned good answers that the answer key does not list), 2 are real.
- Eval is easy because questions were written from the judgements themselves. Treat it as a regression check, not proof of quality. The "~37% NDCG" fine-tuning target is unrealistic; max headroom on Hit@5 is 12 points.
- Found: duplicate chunks fill the top-5 (q030, q032); rare exact words are ignored (q037, "Ranbir"). Evidence for dedup and for the deferred hybrid search.
- Decision: no fixes yet. Next: 50-question RAGAS baseline.
