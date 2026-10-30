"""Ingestion and Chunking Pipeline with Full Provenance Tracking for MedAgentX."""

import sys
import os
import json
import glob
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import structlog
from config.settings import settings

logger = structlog.get_logger()


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Sliding window text chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - chunk_overlap
    return chunks


def process_raw_guidelines(
    raw_dir: str = "data/raw/guidelines",
    output_file: str = "data/processed/chunks.json",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict[str, Any]]:
    """Process raw clinical guidelines and literature into chunks with metadata provenance."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_chunks = []

    json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    logger.info("Found raw guideline files for processing", count=len(json_files))

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
                for doc in documents:
                    doc_id = doc.get("doc_id", "UNKNOWN")
                    content = doc.get("content", "")
                    title = doc.get("title", "")
                    source = doc.get("source", "Clinical Guidelines")
                    pub_date = doc.get("publication_date", "2024-01-01")
                    url = doc.get("url", "")

                    raw_chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    total_chunks = len(raw_chunks)

                    for idx, chunk_str in enumerate(raw_chunks):
                        chunk_record = {
                            "chunk_id": f"{doc_id}_CHK_{idx:03d}",
                            "doc_id": doc_id,
                            "title": title,
                            "source": source,
                            "publication_date": pub_date,
                            "url": url,
                            "chunk_index": idx,
                            "total_chunks": total_chunks,
                            "content": chunk_str,
                            "char_count": len(chunk_str),
                            "preprocessing_version": "1.0.0",
                        }
                        all_chunks.append(chunk_record)
        except Exception as e:
            logger.error("Failed to process guideline file", file=file_path, error=str(e))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    logger.info("Successfully ingested and saved chunks", total_chunks=len(all_chunks), path=output_file)
    return all_chunks


if __name__ == "__main__":
    process_raw_guidelines()
