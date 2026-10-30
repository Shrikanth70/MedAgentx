"""Tests for Phase 3 Baseline Systems (B0, B1, B2)."""

import pytest
from src.baselines.b0_llm_only import B0LLMOnly
from src.baselines.b1_vector_rag import B1VectorRAG
from src.baselines.b2_medical_rag import B2MedicalRAG
from src.baselines.base import SystemResponse


def test_b0_llm_only():
    """Test Baseline B0 LLM-only execution."""
    b0 = B0LLMOnly()
    res = b0.query("What is the initial dual antiplatelet dosage for STEMI?")
    assert isinstance(res, SystemResponse)
    assert res.system_id == "B0_LLM_ONLY"
    assert len(res.evidence_chunks) == 0
    assert res.token_count > 0
    assert res.latency_seconds >= 0.0


def test_b1_vector_rag():
    """Test Baseline B1 Standard Vector RAG execution."""
    b1 = B1VectorRAG()
    res = b1.query("What is the initial dual antiplatelet dosage for STEMI?")
    assert isinstance(res, SystemResponse)
    assert res.system_id == "B1_VECTOR_RAG"
    assert len(res.evidence_chunks) > 0
    assert res.token_count > 0


def test_b2_medical_rag():
    """Test Baseline B2 Advanced Medical RAG execution."""
    b2 = B2MedicalRAG()
    res = b2.query("What is the first-line therapy for HFrEF?")
    assert isinstance(res, SystemResponse)
    assert res.system_id == "B2_MEDICAL_RAG"
    assert len(res.evidence_chunks) > 0
    assert len(res.citations) > 0
