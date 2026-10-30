"""Tests for Phase 2 Data Engineering & Knowledge Indexing Pipeline."""

import os
import json
import pytest
from scripts.ingest import process_raw_guidelines, chunk_text
from scripts.build_embeddings import build_and_index_embeddings, generate_dense_vector
from scripts.build_graph import build_and_index_graph


def test_chunk_text_sliding_window():
    """Test text chunker splitting and overlap logic."""
    sample_text = "A" * 1000
    chunks = chunk_text(sample_text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2
    assert len(chunks[0]) == 500


def test_ingestion_pipeline(tmp_path):
    """Test raw guideline ingestion and chunk generation."""
    chunks = process_raw_guidelines(
        raw_dir="data/raw/guidelines",
        output_file=str(tmp_path / "chunks.json"),
    )
    assert len(chunks) > 0
    first_chunk = chunks[0]
    assert "chunk_id" in first_chunk
    assert "doc_id" in first_chunk
    assert "source" in first_chunk
    assert "preprocessing_version" in first_chunk
    assert first_chunk["preprocessing_version"] == "1.0.0"


def test_generate_dense_vector():
    """Test normalized 768-dim embedding generation."""
    vec = generate_dense_vector("test medical query", vector_dim=768)
    assert len(vec) == 768
    norm = sum(x * x for x in vec) ** 0.5
    assert pytest.approx(norm, 0.001) == 1.0


def test_build_embeddings_pipeline(tmp_path):
    """Test vector embedding caching."""
    chunks_file = tmp_path / "chunks.json"
    cache_file = tmp_path / "vector_cache.json"

    dummy_data = [
        {
            "chunk_id": "TEST_CHK_001",
            "doc_id": "TEST_001",
            "title": "Test Guideline",
            "source": "Test Source",
            "content": "Aspirin is indicated for acute coronary syndrome.",
        }
    ]
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    res = build_and_index_embeddings(
        chunks_file=str(chunks_file),
        cache_file=str(cache_file),
    )
    assert res["status"] == "success"
    assert res["count"] == 1
    assert os.path.exists(cache_file)


def test_build_graph_pipeline(tmp_path):
    """Test Knowledge Graph triples JSON caching."""
    output_file = tmp_path / "graph_triples.json"
    res = build_and_index_graph(output_file=str(output_file))
    assert res["status"] == "success"
    assert os.path.exists(output_file)
    with open(output_file, "r", encoding="utf-8") as f:
        triples = json.load(f)
        assert len(triples) > 0
        assert "subject" in triples[0]
        assert "predicate" in triples[0]
        assert "object" in triples[0]
