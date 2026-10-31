"""Tests for Grounded Finding-Level Query Construction in Hybrid Retriever."""

import pytest
from pydantic import BaseModel
from src.agents.retriever import RetrieverAgent, RetrieverOutput, RetrievalResult
from src.multimodal.contracts import VisualFinding


class MockPatient(BaseModel):
    patient_id: str = "PT-12345"
    age: int = 58
    sex: str = "female"
    chief_complaint: str = "worsening dyspnea and productive cough"
    imaging_modality: str = "Chest X-Ray"


def test_formulate_finding_queries_multiple_findings():
    """Verify formulation of per-finding targeted queries and clinical synthesis query."""
    f1 = VisualFinding(
        label="Lobar Consolidation",
        anatomical_location="Right Middle Lobe",
        laterality="right",
        coding_system="RadLex",
        code="RID49488",
        probability=0.92,
    )
    f2 = VisualFinding(
        label="Pleural Effusion",
        anatomical_location="Right Costophrenic Sulcus",
        laterality="unspecified",
        coding_system="RadLex",
        code="RID5111",
        probability=0.74,
    )

    retriever = RetrieverAgent()
    queries = retriever.formulate_finding_queries(
        visual_findings=[f1, f2],
        chief_complaint="acute dyspnea and fever",
        modality="Chest X-Ray",
    )

    assert len(queries) == 3
    # Finding 1 query incorporates label, location, laterality, modality
    assert "Lobar Consolidation" in queries[0]
    assert "Right Middle Lobe" in queries[0]
    assert "right" in queries[0]
    assert "Chest X-Ray" in queries[0]
    assert "clinical management guideline" in queries[0]

    # Finding 2 query: laterality is unspecified so it should not appear literally
    assert "Pleural Effusion" in queries[1]
    assert "Right Costophrenic Sulcus" in queries[1]
    assert "unspecified" not in queries[1]
    assert "Chest X-Ray" in queries[1]

    # Synthesis query combines all finding labels and chief complaint
    synthesis_q = queries[2]
    assert "Lobar Consolidation" in synthesis_q
    assert "Pleural Effusion" in synthesis_q
    assert "acute dyspnea and fever" in synthesis_q


def test_formulate_finding_queries_empty_and_default():
    """Verify fallback query generation when visual_findings is empty or None."""
    retriever = RetrieverAgent()

    # Empty findings with chief complaint
    q1 = retriever.formulate_finding_queries(
        visual_findings=[],
        chief_complaint="atypical chest pain",
    )
    assert q1 == ["atypical chest pain"]

    # None findings without chief complaint
    q2 = retriever.formulate_finding_queries(visual_findings=None)
    assert q2 == ["Clinical Decision Support Guideline"]


def test_retrieval_with_visual_findings_deduplication_and_provenance():
    """Verify multi-finding retrieval performs deduplication and enriches provenance."""
    f1 = VisualFinding(
        label="Lobar Consolidation",
        anatomical_location="Right Middle Lobe",
        laterality="right",
        code="RID49488",
        probability=0.90,
    )
    f2 = VisualFinding(
        label="Lobar Consolidation",
        anatomical_location="Right Lower Lobe",
        laterality="right",
        code="RID49488",
        probability=0.88,
    )

    retriever = RetrieverAgent()
    result = retriever.retrieve(
        visual_findings=[f1, f2],
        modality="Chest X-Ray",
    )

    assert isinstance(result, RetrievalResult)
    assert isinstance(result, RetrieverOutput)
    assert len(result.queries_executed) >= 2
    assert result.evidence_count == len(result.vector_evidence)

    # Check deduplication
    chunk_ids = [c.get("chunk_id") for c in result.vector_evidence]
    assert len(chunk_ids) == len(set(chunk_ids))

    # Check provenance fields on every returned vector evidence chunk
    for chunk in result.vector_evidence:
        assert "source_id" in chunk and chunk["source_id"]
        assert "publication_year" in chunk and isinstance(chunk["publication_year"], int)
        assert "guideline_body" in chunk and chunk["guideline_body"]
        assert "section" in chunk and chunk["section"]
        assert "url" in chunk and chunk["url"].startswith("http")


def test_retrieval_with_patient_context():
    """Verify patient_context supplies chief_complaint and modality when present."""
    patient = MockPatient()
    f1 = VisualFinding(
        label="Consolidation",
        anatomical_location="Right Lower Lobe",
        laterality="right",
        code="RID49488",
        probability=0.89,
    )

    retriever = RetrieverAgent()
    result = retriever.retrieve(
        visual_findings=[f1],
        patient_context=patient,
    )

    assert len(result.queries_executed) == 2
    # Synthesis query incorporates patient chief complaint
    synthesis_q = result.queries_executed[1]
    assert "Consolidation" in synthesis_q
    assert "worsening dyspnea and productive cough" in synthesis_q
    assert result.evidence_count > 0


def test_backward_compatibility_single_string_query():
    """Verify standard single query invocation works without breaking legacy callers."""
    retriever = RetrieverAgent()
    result = retriever.retrieve(
        query_text="Management of acute coronary syndrome",
        entities=["STEMI", "Aspirin"],
        top_k=3,
    )

    assert isinstance(result, RetrieverOutput)
    assert isinstance(result, RetrievalResult)
    assert len(result.vector_evidence) > 0
    assert result.s_vec > 0.0
    assert result.evidence_count == len(result.vector_evidence)
    assert result.queries_executed == ["Management of acute coronary syndrome"]

    # Provenance fields present even in backward-compatible mode
    first_chunk = result.vector_evidence[0]
    assert "source_id" in first_chunk
    assert "guideline_body" in first_chunk
    assert "publication_year" in first_chunk
    assert "section" in first_chunk
    assert "url" in first_chunk


def test_retrieval_unsupported_query_zero_evidence():
    """Verify unsupported queries yield 0 evidence and 0 scores cleanly."""
    retriever = RetrieverAgent()
    result = retriever.retrieve(query_text="unsupported_query syndrome x")

    assert result.evidence_count == 0
    assert len(result.vector_evidence) == 0
    assert result.s_vec == 0.0
    assert result.s_graph == 0.0
