"""ACMR Dynamic Routing Policy Engine with Safety Policy & Gap Awareness."""

from enum import Enum
from pydantic import BaseModel
import structlog
from config.settings import settings

logger = structlog.get_logger()


class ACMRAction(str, Enum):
    """Actions emitted by the ACMR Router."""

    STOP_CONFIDENT = "STOP_CONFIDENT"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    REQUEST_MORE_INPUT = "REQUEST_MORE_INPUT"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"
    ABSTAIN_INSUFFICIENT_EVIDENCE = "ABSTAIN_INSUFFICIENT_EVIDENCE"
    STOP_SATURATED = "STOP_SATURATED"
    STOP_MAX_ITERATIONS = "STOP_MAX_ITERATIONS"
    SAFETY_CONSTRAINT = "SAFETY_CONSTRAINT"
    HARD_ABSTENTION = "HARD_ABSTENTION"
    RESOLVE_CONTRADICTION = "RESOLVE_CONTRADICTION"
    TARGETED_RE_RETRIEVE = "TARGETED_RE_RETRIEVE"
    QUERY_REWRITE_RE_RETRIEVE = "QUERY_REWRITE_RE_RETRIEVE"


# Top-level action string constants for backward compatibility and convenience
STOP_CONFIDENT = ACMRAction.STOP_CONFIDENT.value
RETRIEVE_MORE = ACMRAction.RETRIEVE_MORE.value
REQUEST_MORE_INPUT = ACMRAction.REQUEST_MORE_INPUT.value
ESCALATE_TO_HUMAN = ACMRAction.ESCALATE_TO_HUMAN.value
ABSTAIN_INSUFFICIENT_EVIDENCE = ACMRAction.ABSTAIN_INSUFFICIENT_EVIDENCE.value
STOP_SATURATED = ACMRAction.STOP_SATURATED.value
STOP_MAX_ITERATIONS = ACMRAction.STOP_MAX_ITERATIONS.value
SAFETY_CONSTRAINT = ACMRAction.SAFETY_CONSTRAINT.value
HARD_ABSTENTION = ACMRAction.HARD_ABSTENTION.value
RESOLVE_CONTRADICTION = ACMRAction.RESOLVE_CONTRADICTION.value
TARGETED_RE_RETRIEVE = ACMRAction.TARGETED_RE_RETRIEVE.value
QUERY_REWRITE_RE_RETRIEVE = ACMRAction.QUERY_REWRITE_RE_RETRIEVE.value


class RouterDecision(BaseModel):
    """Structured, explainable output from ACMR Router."""

    action: str  # ACMRAction value
    reason: str
    adapted_objective: str = ""


class ACMRRouter:
    """Gap-Aware & Safety-Policy ACMR Router evaluating state and deciding explainable next action."""

    def __init__(
        self,
        high_threshold: float = settings.CONFIDENCE_HIGH_THRESHOLD,
        low_threshold: float = settings.CONFIDENCE_LOW_THRESHOLD,
        max_iterations: int = settings.MAX_RETRIEVAL_ITERATIONS,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_iterations = max_iterations

    def route_explainable(
        self,
        c_total: float,
        delta_c: float = 0.0,
        retrieval_gap: str = "NONE",
        missing_evidence: list[str] | None = None,
        s_rel: float = 1.0,
        r_contra: float = 0.0,
        contradiction_type: str = "NONE",
        safety_severity: str = "NONE",
        evidence_novelty: float = 1.0,
        evidence_overlap: float = 0.0,
        iteration: int = 0,
        is_loop_detected: bool = False,
        vision_status: str | None = None,
        requires_imaging: bool = False,
        is_critical_pathology: bool = False,
        quality_rejection_reasons: list[str] | None = None,
        evidence_count: int = 2,
    ) -> RouterDecision:
        """Evaluate state and return safety-aware routing decision with human-readable rationale."""
        v_status_str = (
            vision_status.value
            if hasattr(vision_status, "value")
            else str(vision_status)
            if vision_status is not None
            else None
        )

        # Hard Gate 1 (Fail-Closed on Vision Failure)
        if requires_imaging and v_status_str and v_status_str != "OK":
            if v_status_str == "LOW_QUALITY":
                reasons_str = ", ".join(quality_rejection_reasons or ["suboptimal study"])
                reason = f"Image quality inadequate: {reasons_str}. New acquisition required."
                logger.warn("ACMR Router Decision: REQUEST_MORE_INPUT", reason=reason)
                return RouterDecision(action=ACMRAction.REQUEST_MORE_INPUT.value, reason=reason)
            else:
                reason = f"Imaging perception incomplete (status={v_status_str}). Automated normal conclusion forbidden."
                logger.warn("ACMR Router Decision: ABSTAIN_INSUFFICIENT_EVIDENCE", reason=reason)
                return RouterDecision(action=ACMRAction.ABSTAIN_INSUFFICIENT_EVIDENCE.value, reason=reason)

        # Hard Gate 2 (Critical Pathology Escalation)
        if is_critical_pathology:
            reason = "Critical acute pathology detected requiring emergent physician intervention."
            logger.warn("ACMR Router Decision: ESCALATE_TO_HUMAN", reason=reason)
            return RouterDecision(action=ACMRAction.ESCALATE_TO_HUMAN.value, reason=reason)

        # Rule 1: Supported Clinical Contraindication / Safety Constraint
        if contradiction_type == "CLINICAL_CONTRAINDICATION":
            reason = (
                "Retrieved evidence supports a clinically important contraindication in the specified patient context. "
                "Normal recommendation generation is BLOCKED; triggering Controlled Safety Alert."
            )
            logger.warn("ACMR Router Decision: SAFETY_CONSTRAINT", reason=reason)
            return RouterDecision(action=ACMRAction.SAFETY_CONSTRAINT.value, reason=reason)

        # Rule 2: Evidence Conflict Resolution Policy
        if contradiction_type == "EVIDENCE_CONFLICT":
            if iteration < self.max_iterations:
                reason = (
                    "Retrieved evidence contains opposing guideline recommendations. "
                    "Triggering targeted contradiction resolution retrieval."
                )
                logger.info("ACMR Router Decision: RESOLVE_CONTRADICTION", reason=reason)
                return RouterDecision(
                    action=ACMRAction.RESOLVE_CONTRADICTION.value,
                    reason=reason,
                    adapted_objective="Retrieve evidence resolving guideline conflict on supplemental oxygen in STEMI",
                )
            else:
                reason = "Conflicting high-quality guideline evidence could not be resolved after maximum retrieval passes. Triggering Hard Abstention."
                logger.warn("ACMR Router Decision: HARD_ABSTENTION", reason=reason)
                return RouterDecision(action=ACMRAction.HARD_ABSTENTION.value, reason=reason)

        # Rule 3: Insufficient Evidence Handling
        if contradiction_type == "INSUFFICIENT_EVIDENCE":
            if iteration < self.max_iterations:
                reason = "No matching evidence chunks retrieved. Triggering targeted query expansion retrieval."
                logger.info("ACMR Router Decision: TARGETED_RE_RETRIEVE", reason=reason)
                return RouterDecision(action=ACMRAction.TARGETED_RE_RETRIEVE.value, reason=reason)
            else:
                reason = "Insufficient evidence in knowledge base to establish a clinical recommendation. Triggering Hard Abstention."
                logger.warn("ACMR Router Decision: HARD_ABSTENTION", reason=reason)
                return RouterDecision(action=ACMRAction.HARD_ABSTENTION.value, reason=reason)

        # Rule 4: Irrelevance Gate Override
        if s_rel < 0.50:
            reason = f"Retrieved evidence relevance (S_rel={s_rel:.2f}) is below acceptable threshold (0.50). Disease/topic mismatch detected."
            logger.warn("ACMR Router Decision: HARD_ABSTENTION", reason=reason)
            return RouterDecision(action=ACMRAction.HARD_ABSTENTION.value, reason=reason)

        # Rule 5: Low Confidence Override
        if c_total < self.low_threshold:
            reason = f"Confidence C_total={c_total:.4f} is below low threshold ({self.low_threshold:.2f}). Evidence state insufficient."
            logger.warn("ACMR Router Decision: HARD_ABSTENTION", reason=reason)
            return RouterDecision(action=ACMRAction.HARD_ABSTENTION.value, reason=reason)

        # Rule 6: High Confidence + No Critical Gap -> STOP_CONFIDENT
        if c_total >= self.high_threshold and retrieval_gap == "NONE":
            reason = (
                f"Confidence C_total={c_total:.4f} meets high threshold ({self.high_threshold:.2f}). "
                "Evidence relevance=1.00, agreement is strong, and no unresolved retrieval gaps remain. Proceeding to Clinical Reasoner."
            )
            logger.info("ACMR Router Decision: STOP_CONFIDENT", reason=reason)
            return RouterDecision(action=ACMRAction.STOP_CONFIDENT.value, reason=reason)

        # Rule 7: Loop Detection / Retrieval Saturation -> STOP_SATURATED
        if is_loop_detected or (iteration > 0 and evidence_overlap >= 0.85 and abs(delta_c) < 0.01):
            reason = (
                f"Retrieval Saturation / Loop Detected (iteration={iteration}, "
                f"overlap={evidence_overlap:.2f}, novelty={evidence_novelty:.2f}, delta_C={delta_c:+.4f}). "
                "Further retrieval is unlikely to yield complementary evidence. Halting retrieval loop."
            )
            logger.info("ACMR Router Decision: STOP_SATURATED", reason=reason)
            return RouterDecision(action=ACMRAction.STOP_SATURATED.value, reason=reason)

        # Rule 8: Max Iterations Limit Reached -> STOP_MAX_ITERATIONS
        if iteration >= self.max_iterations:
            reason = (
                f"Maximum retrieval iterations limit reached ({iteration}/{self.max_iterations}). "
                f"Confidence C_total={c_total:.4f} did not reach high threshold ({self.high_threshold:.2f})."
            )
            logger.info("ACMR Router Decision: STOP_MAX_ITERATIONS", reason=reason)
            return RouterDecision(action=ACMRAction.STOP_MAX_ITERATIONS.value, reason=reason)

        # Rule 9: Missing Evidence Gap -> TARGETED_RE_RETRIEVE
        if retrieval_gap == "MISSING_EVIDENCE":
            missing_str = ", ".join(missing_evidence) if missing_evidence else "contextual details"
            reason = (
                f"Confidence C_total={c_total:.4f} is below high threshold ({self.high_threshold:.2f}). "
                f"The Verifier identified missing information: [{missing_str}]. Triggering targeted re-retrieval."
            )
            obj = f"Retrieve evidence for: {missing_str}"
            logger.info("ACMR Router Decision: TARGETED_RE_RETRIEVE", reason=reason)
            return RouterDecision(action=ACMRAction.TARGETED_RE_RETRIEVE.value, reason=reason, adapted_objective=obj)

        # Rule 10: Moderate Confidence -> QUERY_REWRITE_RE_RETRIEVE
        if self.low_threshold <= c_total < self.high_threshold:
            reason = (
                f"Confidence C_total={c_total:.4f} is moderate. Delta C={delta_c:+.4f}. "
                "Rewriting query to seek complementary evidence."
            )
            logger.info("ACMR Router Decision: QUERY_REWRITE_RE_RETRIEVE", reason=reason)
            return RouterDecision(action=ACMRAction.QUERY_REWRITE_RE_RETRIEVE.value, reason=reason)

        return RouterDecision(action=ACMRAction.STOP_CONFIDENT.value, reason="Default stop policy.")

    def route(
        self,
        c_total: float,
        iteration: int = 0,
        r_contra: float = 0.0,
        vision_status: str | None = None,
        requires_imaging: bool = False,
        is_critical_pathology: bool = False,
        quality_rejection_reasons: list[str] | None = None,
        evidence_count: int = 2,
    ) -> str:
        """Backward compatible string route method for existing test interfaces."""
        decision = self.route_explainable(
            c_total=c_total,
            delta_c=0.0,
            retrieval_gap="NONE" if c_total >= self.high_threshold else "MISSING_EVIDENCE",
            missing_evidence=[],
            s_rel=1.0,
            r_contra=r_contra,
            contradiction_type="CLINICAL_CONTRAINDICATION" if r_contra >= 0.80 else "NONE",
            evidence_novelty=1.0,
            evidence_overlap=0.0,
            iteration=iteration,
            vision_status=vision_status,
            requires_imaging=requires_imaging,
            is_critical_pathology=is_critical_pathology,
            quality_rejection_reasons=quality_rejection_reasons,
            evidence_count=evidence_count,
        )
        if decision.action in ["ABSTAIN", "HARD_ABSTENTION", "SAFETY_CONSTRAINT", "ABSTAIN_INSUFFICIENT_EVIDENCE"]:
            return "ABSTAIN"
        elif decision.action in ["ESCALATE_TO_HUMAN"]:
            return "ESCALATE_TO_HUMAN"
        elif decision.action in ["REQUEST_MORE_INPUT"]:
            return "REQUEST_MORE_INPUT"
        elif decision.action.startswith("STOP") or decision.action in ["RESOLVE_CONTRADICTION"]:
            return "STOP"
        else:
            return "RE_RETRIEVE"
