"""Unit tests for ACMR Router Hard-Gates, Safety Policies, and Confidence Penalties."""

import pytest
from src.orchestrator.confidence import ConfidenceCalculator, CompositeConfidenceCalculator
from src.orchestrator.router import (
    ACMRRouter,
    ACMRAction,
    RouterDecision,
    STOP_CONFIDENT,
    RETRIEVE_MORE,
    REQUEST_MORE_INPUT,
    ESCALATE_TO_HUMAN,
    ABSTAIN_INSUFFICIENT_EVIDENCE,
)


class TestRouterHardGates:
    """Test safety hard-gates in ACMRRouter.route_explainable."""

    @pytest.fixture
    def router(self):
        return ACMRRouter(high_threshold=0.80, low_threshold=0.50, max_iterations=2)

    def test_vision_timeout_triggers_abstention(self, router):
        """Invariant I1: Vision timeout must fail-closed to ABSTAIN_INSUFFICIENT_EVIDENCE."""
        decision = router.route_explainable(
            c_total=0.95,
            requires_imaging=True,
            vision_status="TIMEOUT",
        )
        assert decision.action == ABSTAIN_INSUFFICIENT_EVIDENCE
        assert "Imaging perception incomplete" in decision.reason
        assert "TIMEOUT" in decision.reason

    def test_vision_failed_triggers_abstention(self, router):
        """Vision perception failure must fail-closed."""
        decision = router.route_explainable(
            c_total=0.90,
            requires_imaging=True,
            vision_status="FAILED",
        )
        assert decision.action == ABSTAIN_INSUFFICIENT_EVIDENCE
        assert "Imaging perception incomplete" in decision.reason
        assert "FAILED" in decision.reason

    def test_vision_unsupported_triggers_abstention(self, router):
        """Unsupported modality/study must fail-closed."""
        decision = router.route_explainable(
            c_total=0.90,
            requires_imaging=True,
            vision_status="UNSUPPORTED",
        )
        assert decision.action == ABSTAIN_INSUFFICIENT_EVIDENCE
        assert "UNSUPPORTED" in decision.reason

    def test_low_quality_triggers_request_more_input(self, router):
        """Low quality study triggers REQUEST_MORE_INPUT with specific rejection reasons."""
        decision = router.route_explainable(
            c_total=0.88,
            requires_imaging=True,
            vision_status="LOW_QUALITY",
            quality_rejection_reasons=["severe motion artifact", "severe underexposure"],
        )
        assert decision.action == REQUEST_MORE_INPUT
        assert "Image quality inadequate" in decision.reason
        assert "severe motion artifact" in decision.reason
        assert "severe underexposure" in decision.reason

    def test_low_quality_default_reason_if_none_provided(self, router):
        """Low quality study handles empty/None rejection reasons gracefully."""
        decision = router.route_explainable(
            c_total=0.88,
            requires_imaging=True,
            vision_status="LOW_QUALITY",
            quality_rejection_reasons=None,
        )
        assert decision.action == REQUEST_MORE_INPUT
        assert "suboptimal study" in decision.reason

    def test_critical_pathology_triggers_escalation(self, router):
        """Hard Gate 2: Critical pathology detected triggers ESCALATE_TO_HUMAN."""
        decision = router.route_explainable(
            c_total=0.92,
            requires_imaging=True,
            vision_status="OK",
            is_critical_pathology=True,
        )
        assert decision.action == ESCALATE_TO_HUMAN
        assert "Critical acute pathology detected" in decision.reason

    def test_normal_case_with_ok_vision_and_high_confidence(self, router):
        """Normal case with OK vision and high confidence routes to STOP_CONFIDENT."""
        decision = router.route_explainable(
            c_total=0.88,
            requires_imaging=True,
            vision_status="OK",
            is_critical_pathology=False,
            retrieval_gap="NONE",
        )
        assert decision.action == STOP_CONFIDENT
        assert "meets high threshold" in decision.reason

    def test_requires_imaging_false_ignores_vision_status(self, router):
        """When imaging is not required, vision_status does not block STOP_CONFIDENT."""
        decision = router.route_explainable(
            c_total=0.88,
            requires_imaging=False,
            vision_status="TIMEOUT",
            retrieval_gap="NONE",
        )
        assert decision.action == STOP_CONFIDENT

    def test_route_string_method_compatibility(self, router):
        """Test backward-compatible route() string method with new safety flags."""
        assert router.route(c_total=0.90, requires_imaging=True, vision_status="TIMEOUT") == "ABSTAIN"
        assert router.route(c_total=0.90, requires_imaging=True, vision_status="LOW_QUALITY") == "REQUEST_MORE_INPUT"
        assert router.route(c_total=0.90, is_critical_pathology=True) == "ESCALATE_TO_HUMAN"
        assert router.route(c_total=0.85, requires_imaging=True, vision_status="OK") == "STOP"


class TestConfidencePenalties:
    """Test penalty terms in ConfidenceCalculator."""

    def test_baseline_confidence_no_penalties(self):
        """When vision_status='OK', vision_uncertainty=0, evidence_count>=2, no penalty applies."""
        score = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90,
            s_graph=1.0,
            v_agree=0.85,
            r_contra=0.0,
            s_rel=1.0,
            completeness=1.0,
            novelty=1.0,
            vision_status="OK",
            vision_uncertainty=0.0,
            evidence_count=2,
        )
        # 0.30*0.90 + 0.20*1.0 + 0.20*0.85 + 0.20*1.0 + 0.10*1.0 = 0.27 + 0.20 + 0.17 + 0.20 + 0.10 = 0.94
        assert score == 0.94

    def test_vision_status_penalty(self):
        """When vision_status != 'OK', a 0.25 penalty is deducted."""
        score_ok = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, vision_status="OK"
        )
        score_timeout = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, vision_status="TIMEOUT"
        )
        assert round(score_ok - score_timeout, 4) == 0.25

    def test_vision_uncertainty_penalty(self):
        """When vision_uncertainty > 0, 0.15 * clamp(uncertainty, 0, 1) penalty is deducted."""
        score_zero_unc = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, vision_uncertainty=0.0
        )
        score_unc_half = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, vision_uncertainty=0.5
        )
        # Expected penalty: 0.15 * 0.5 = 0.075
        assert round(score_zero_unc - score_unc_half, 4) == 0.075

    def test_vision_uncertainty_clamping(self):
        """Uncertainty values > 1.0 are clamped to 1.0 (max penalty 0.15)."""
        score_base = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, vision_uncertainty=0.0
        )
        score_extreme = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, vision_uncertainty=2.5
        )
        assert round(score_base - score_extreme, 4) == 0.15

    def test_thin_evidence_count_penalty(self):
        """When evidence_count < 2, a 0.10 penalty is deducted."""
        score_two_chunks = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, evidence_count=2
        )
        score_one_chunk = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.90, s_graph=1.0, v_agree=0.85, r_contra=0.0, evidence_count=1
        )
        assert round(score_two_chunks - score_one_chunk, 4) == 0.10

    def test_combined_penalties(self):
        """Combined vision failure, uncertainty, and thin evidence sum additively and clamp at 0.0."""
        score = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=0.50,
            s_graph=0.50,
            v_agree=0.50,
            r_contra=0.20,
            vision_status="FAILED",        # -0.25
            vision_uncertainty=1.0,       # -0.15
            evidence_count=0,             # -0.10
        )
        # raw: 0.3*0.5 + 0.2*0.5 + 0.2*0.5 + 0.2*1.0 + 0.1*1.0 - 0.15*0.2 = 0.15 + 0.10 + 0.10 + 0.20 + 0.10 - 0.03 = 0.62
        # penalty: 0.25 + 0.15 + 0.10 = 0.50
        # total: 0.62 - 0.50 = 0.12
        assert score == 0.12

    def test_missing_scores_not_defaulted_to_one(self):
        """Missing or None scores evaluate to 0.0, not 1.0."""
        score_none = ConfidenceCalculator.calculate_composite_confidence(
            s_vec=None,
            s_graph=None,
            v_agree=None,
            r_contra=0.0,
            completeness=None,
            novelty=None,
            s_rel=None,
        )
        assert score_none == 0.0

    def test_backward_compatibility_alias(self):
        """CompositeConfidenceCalculator alias must support new arguments."""
        score = CompositeConfidenceCalculator.calculate(
            s_vec=0.90,
            s_graph=1.0,
            v_agree=0.85,
            r_contra=0.0,
            vision_status="OK",
            vision_uncertainty=0.0,
            evidence_count=2,
        )
        assert score == 0.94
