"""Tests for Phase 7 ACMR Orchestrator & MedAgentX State Machine."""

import pytest
from src.orchestrator.confidence import CompositeConfidenceCalculator
from src.orchestrator.router import ACMRRouter
from src.orchestrator.system import MedAgentXSystem
from src.baselines.base import SystemResponse


def test_confidence_calculator():
    """Test mathematical C_total score calculation."""
    calc = CompositeConfidenceCalculator()
    # High confidence test
    score = calc.calculate(s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0)
    assert 0.70 <= score <= 1.0

    # Contradiction penalty test
    score_contra = calc.calculate(s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.80)
    assert score_contra < score


def test_acmr_router_decisions():
    """Test ACMR dynamic routing policy rules."""
    router = ACMRRouter(high_threshold=0.80, low_threshold=0.50, max_iterations=2)

    # Rule 1: High confidence -> STOP
    assert router.route(c_total=0.85, iteration=0, r_contra=0.0) == "STOP"

    # Rule 2: Moderate confidence -> RE_RETRIEVE
    assert router.route(c_total=0.65, iteration=0, r_contra=0.0) == "RE_RETRIEVE"

    # Rule 3: Clinical contradiction -> ABSTAIN
    assert router.route(c_total=0.75, iteration=0, r_contra=0.80) == "ABSTAIN"

    # Rule 4: Low confidence -> ABSTAIN
    assert router.route(c_total=0.30, iteration=0, r_contra=0.0) == "ABSTAIN"


def test_medagentx_system_execution():
    """Test full MedAgentX ACMR System query execution."""
    system = MedAgentXSystem()
    res = system.query("What is the initial dual antiplatelet dosage for STEMI?")
    assert isinstance(res, SystemResponse)
    assert res.system_id == "PROPOSED_MEDAGENTX"
    assert "c_total" in res.metadata
    assert "route_decision" in res.metadata
    assert "trace" in res.metadata
    assert res.token_count > 0


def test_medagentx_system_abstention():
    """Test MedAgentX system safety abstention on contradiction query."""
    system = MedAgentXSystem()
    res = system.query("Administer sublingual nitroglycerin for Right Ventricular Infarction in STEMI")
    assert isinstance(res, SystemResponse)
    route = res.metadata.get("route_decision", "")
    assert route in ["ABSTAIN", "SAFETY_CONSTRAINT", "HARD_ABSTENTION"] or "ABSTAIN" in res.answer or "SAFETY ALERT" in res.answer
