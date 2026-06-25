from datasets import load_dataset  

def ingest_from_hf(num_docs=200):
    ds = load_dataset("pile-of-law/pile-of-law", "indiankanoon", 
                       split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= num_docs:
            break
        # save as JSON to data/processed/
        save_processed({
            "doc_id": str(i),
            "text": row["text"],
            "url": row.get("url", ""),
            "scraped_at": row.get("created_timestamp", ""),
        })