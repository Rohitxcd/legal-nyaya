# Nyaya - Legal Document Processing Pipeline

## Project Overview
A document ingestion and processing system for Indian legal documents from the Pile-of-Law dataset. The system scrapes and processes legal case documents, extracts embeddings, and enables semantic search capabilities using sentence transformers and FAISS.

## Current Phase
**EARLY DEVELOPMENT** - Infrastructure and Core Components Setup

## Project Status

### ✅ Completed
- Project structure initialized
- Python environment setup with venv
- Dependencies defined (requests, beautifulsoup4, sentence-transformers, faiss-cpu, datasets)
- Data ingestion logic drafted in `data/scraper.py`
- Document schema defined in `data/processed/doc_001.json` (fields: doc_id, title, court, year, url, text)

### 🔄 In Progress
- Setting up data ingestion pipeline
- Implementing document processing workflow

### ⏳ Pending
- Complete `src/ingest.py` implementation
- Add document embedding generation
- Set up FAISS vector store
- Build semantic search interface
- Add error handling and logging
- Write unit tests

## Technical Stack
- **Data Source**: Hugging Face (pile-of-law/indiankanoon dataset)
- **NLP**: sentence-transformers (embeddings)
- **Vector Store**: FAISS (semantic search)
- **Data Processing**: datasets library
- **Web Scraping**: BeautifulSoup4

## Key Files
- `data/scraper.py` - Document ingestion from HF datasets (partially implemented)
- `src/ingest.py` - Main processing pipeline (EMPTY - needs implementation)
- `data/processed/doc_001.json` - Sample document schema with fields: doc_id, title, court, year, url, text
- `requirements.txt` - Python dependencies

## Next Steps
1. Complete `src/ingest.py` implementation with document loading and processing
2. Populate `data/processed/` with actual documents from HF dataset
3. Add document embedding generation using sentence-transformers
4. Initialize and populate FAISS vector store
5. Build query interface for semantic search
6. Add error handling and logging
7. Write unit tests

---
*Last Updated: 2026-06-24 (Updated with current project state)*
