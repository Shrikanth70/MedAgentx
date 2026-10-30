"""Tests for Phase 4 Hybrid Retrieval Engine (Qdrant + Neo4j)."""

import pytest
from src.knowledge.hybrid_retriever import HybridRetriever


def test_hybrid_search_vector():
    """Test vector search score calculation."""
    retriever = HybridRetriever()
    results = retriever.search_vector("ST-elevation myocardial infarction STEMI aspirin", top_k=3)
    assert len(results) > 0
    assert "content" in results[0]
    assert "vector_score" in results[0]


def test_hybrid_search_graph():
    """Test Knowledge Graph triple search."""
    retriever = HybridRetriever()
    triples = retriever.search_graph(["STEMI", "Aspirin"])
    assert len(triples) > 0
    assert "subject" in triples[0]


def test_retrieve_hybrid_fusion():
    """Test bi-modal hybrid retrieval score fusion."""
    retriever = HybridRetriever()
    res = retriever.retrieve_hybrid(
        query_text="Management of STEMI and dual antiplatelet therapy",
        entities=["STEMI", "Aspirin"],
        top_k=3,
        alpha=0.6,
    )
    assert "s_vec" in res
    assert "s_graph" in res
    assert "hybrid_score" in res
    assert res["s_vec"] > 0.0
    assert res["s_graph"] > 0.0
    assert res["hybrid_score"] > 0.0
