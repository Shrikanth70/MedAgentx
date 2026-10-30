"""Tests for Phase 5 Core Specialized Agent Modules."""

import pytest
from src.agents.planner import PlannerAgent, PlannerOutput
from src.agents.retriever import RetrieverAgent, RetrieverOutput
from src.agents.verifier import VerifierAgent, VerifierOutput
from src.agents.reasoner import ReasonerAgent, ReasonerOutput


def test_planner_agent():
    """Test Agent 1 Planner & Reformulator."""
    planner = PlannerAgent()
    out = planner.plan("What is the initial dual antiplatelet therapy for STEMI?")
    assert isinstance(out, PlannerOutput)
    assert len(out.decomposed_subqueries) == 3
    assert len(out.extracted_entities) > 0


def test_retriever_agent():
    """Test Agent 2 Hybrid Retriever Engine."""
    retriever = RetrieverAgent()
    out = retriever.retrieve(
        query_text="Management of acute coronary syndrome",
        entities=["STEMI", "Aspirin"],
        top_k=3,
    )
    assert isinstance(out, RetrieverOutput)
    assert len(out.vector_evidence) > 0
    assert out.s_vec > 0.0


def test_verifier_agent():
    """Test Agent 3 Evidence Verifier & Critic."""
    verifier = VerifierAgent()
    out = verifier.verify(
        query_text="Is sublingual nitroglycerin safe in STEMI?",
        vector_evidence=[{"content": "Aspirin 325 mg initial loading dose."}],
        graph_evidence=[{"subject": "STEMI"}],
    )
    assert isinstance(out, VerifierOutput)
    assert out.v_agree > 0.0
    assert out.r_contra >= 0.0


def test_verifier_agent_contradiction():
    """Test Agent 3 contradiction detection for Nitroglycerin + Right Ventricular Infarction."""
    verifier = VerifierAgent()
    out = verifier.verify(
        query_text="Administer sublingual nitroglycerin for Right Ventricular Infarction",
        vector_evidence=[{"content": "nitroglycerin is contraindicated in right ventricular infarction."}],
        graph_evidence=[],
    )
    assert len(out.detected_contradictions) > 0
    assert out.r_contra > 0.50


def test_reasoner_agent():
    """Test Agent 4 Clinical Reasoner & Synthesizer."""
    reasoner = ReasonerAgent()
    out = reasoner.synthesize(
        query_text="Management of STEMI",
        verified_evidence=[{"source": "AHA Guidelines", "content": "High-dose aspirin 325 mg."}],
        detected_contradictions=[],
        c_total=0.85,
    )
    assert isinstance(out, ReasonerOutput)
    assert len(out.clinical_report) > 0
    assert len(out.citations) > 0
