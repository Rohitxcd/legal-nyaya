# Nyaya - Legal Document Processing Pipeline

## Project Overview
A document ingestion and processing system for Indian legal documents from the Pile-of-Law dataset. The system collects legal case documents, normalizes them into JSON, chunks the text, generates embeddings, builds a FAISS index, and supports semantic search.

## Current Phase
**PROTOTYPE STAGE** - Core ingestion, chunking, embedding, and search pieces are in place

## Project Status

### ✅ Completed
- Project structure initialized
- Python environment setup with venv
- Dependencies defined (requests, beautifulsoup4, sentence-transformers, faiss-cpu, datasets)
- Dataset ingestion draft added in `src/ingest/scraper.py`
- JSON export helper created in `src/ingest/save_json.py`
- Document processing samples created in `data/processed/`
- Chunking pipeline implemented in `src/ingest/chunker.py`
- Embedding and FAISS index builder implemented in `src/ingest/embedder.py`
- Semantic query script implemented in `src/ingest/query.py`
- Sample index artifacts created in `data/index/`

### 🔄 In Progress
- Validating the end-to-end pipeline from raw text to search results
- Cleaning up ad-hoc document export and making JSON generation reusable
- Improving path handling and error handling across ingest scripts

### ⏳ Pending
- Add a single orchestrated pipeline entry point
- Replace one-off export code with a reusable document writer
- Add logging and structured error handling
- Add unit tests for chunking, embedding, and query behavior
- Add integration tests for the ingestion-to-index flow
- Improve documentation and usage examples

## Technical Stack
- **Data Source**: Hugging Face (pile-of-law/indiankanoon dataset)
- **NLP**: sentence-transformers (embeddings)
- **Vector Store**: FAISS (semantic search)
- **Data Processing**: datasets library
- **Web Scraping**: BeautifulSoup4

## Key Files
- `src/ingest/scraper.py` - Hugging Face dataset ingestion draft
- `src/ingest/save_json.py` - One-off helper used to write sample processed documents
- `src/ingest/chunker.py` - Splits processed judgments into overlapping chunks
- `src/ingest/embedder.py` - Builds embeddings and writes the FAISS index
- `src/ingest/query.py` - Runs semantic search against the index
- `data/processed/` - Processed JSON documents
- `data/chunks/` - Chunked document JSON files
- `data/index/faiss.index` - FAISS vector index
- `requirements.txt` - Python dependencies

## Next Steps
1. Add a single pipeline script that runs ingest, chunk, embed, and query preparation in order
2. Harden the scraper and JSON writer so processed documents are created consistently
3. Add logging, validation, and error handling around file reads and dataset records
4. Add tests for chunking, embedding metadata, and search result formatting
5. Document how to regenerate processed docs, chunks, and the FAISS index

---
*Last Updated: 2026-06-25 (Updated with current project state)*
