"""Composite Evidence Confidence Calculator ($C_{total}$) for MedAgentX."""

import structlog

logger = structlog.get_logger()


class ConfidenceCalculator:
    """Calculates Composite Evidence Confidence Score ($C_{total}$) with Relevance & Completeness Weighting."""

    @staticmethod
    def calculate_composite_confidence(
        s_vec: float | None,
        s_graph: float | None,
        v_agree: float | None,
        r_contra: float | None,
        s_rel: float | None = 1.0,
        completeness: float | None = 1.0,
        novelty: float | None = 1.0,
        contradiction_type: str = "NONE",
        vision_status: str = "OK",
        vision_uncertainty: float = 0.0,
        evidence_count: int = 2,
    ) -> float:
        """Calculate composite confidence score ($C_{total}$) gated by Evidence Relevance ($S_{rel}$) & Completeness.
        
        Formula:
            If CLINICAL_CONTRAINDICATION: Confidence measures strong evidence grounding for safety constraint (r_contra not subtracted from C_total).
            Else: C_total = 0.30*(S_vec*S_rel) + 0.20*(S_graph*S_rel) + 0.20*V_agree + 0.20*completeness + 0.10*novelty - 0.15*R_contra - Pi_penalty
            
            Where Pi_penalty:
                + 0.25 if vision_status != 'OK'
                + 0.15 * clamp(vision_uncertainty, 0.0, 1.0) if vision_uncertainty > 0.0
                + 0.10 if evidence_count < 2
        """
        # Ensure missing/None values evaluate to 0.0, not defaulted to 1.0
        safe_s_vec = 0.0 if s_vec is None else float(s_vec)
        safe_s_graph = 0.0 if s_graph is None else float(s_graph)
        safe_v_agree = 0.0 if v_agree is None else float(v_agree)
        safe_r_contra = 0.0 if r_contra is None else float(r_contra)
        safe_s_rel = 0.0 if s_rel is None else float(s_rel)
        safe_completeness = 0.0 if completeness is None else float(completeness)
        safe_novelty = 0.0 if novelty is None else float(novelty)
        safe_vision_unc = 0.0 if vision_uncertainty is None else float(vision_uncertainty)
        safe_ev_count = 0 if evidence_count is None else int(evidence_count)

        v_status_str = (
            vision_status.value
            if hasattr(vision_status, "value")
            else str(vision_status)
            if vision_status is not None
            else "OK"
        )

        effective_s_vec = safe_s_vec * safe_s_rel
        effective_s_graph = safe_s_graph * safe_s_rel

        contra_penalty = 0.15 * safe_r_contra if contradiction_type != "CLINICAL_CONTRAINDICATION" else 0.0

        raw_score = (
            0.30 * effective_s_vec
            + 0.20 * effective_s_graph
            + 0.20 * safe_v_agree
            + 0.20 * safe_completeness
            + 0.10 * safe_novelty
            - contra_penalty
        )

        # Calculate penalty term Pi_penalty
        pi_penalty = 0.0
        if v_status_str != "OK":
            pi_penalty += 0.25
        if safe_vision_unc > 0.0:
            clamped_unc = max(0.0, min(1.0, safe_vision_unc))
            pi_penalty += 0.15 * clamped_unc
        if safe_ev_count < 2:
            pi_penalty += 0.10

        c_total = max(0.0, min(1.0, raw_score - pi_penalty))
        rounded_c_total = round(c_total, 4)

        logger.info(
            "Calculated Composite Confidence",
            c_total=rounded_c_total,
            raw_score=raw_score,
            pi_penalty=pi_penalty,
            s_vec=safe_s_vec,
            s_graph=safe_s_graph,
            v_agree=safe_v_agree,
            r_contra=safe_r_contra,
            s_rel=safe_s_rel,
            completeness=safe_completeness,
            novelty=safe_novelty,
            contradiction_type=contradiction_type,
            vision_status=v_status_str,
            vision_uncertainty=safe_vision_unc,
            evidence_count=safe_ev_count,
        )
        return rounded_c_total

    @classmethod
    def calculate(
        cls,
        s_vec: float | None,
        s_graph: float | None,
        v_agree: float | None,
        r_contra: float | None,
        s_rel: float | None = 1.0,
        completeness: float | None = 1.0,
        novelty: float | None = 1.0,
        contradiction_type: str = "NONE",
        vision_status: str = "OK",
        vision_uncertainty: float = 0.0,
        evidence_count: int = 2,
    ) -> float:
        """Alias method for test suite compatibility."""
        return cls.calculate_composite_confidence(
            s_vec=s_vec,
            s_graph=s_graph,
            v_agree=v_agree,
            r_contra=r_contra,
            s_rel=s_rel,
            completeness=completeness,
            novelty=novelty,
            contradiction_type=contradiction_type,
            vision_status=vision_status,
            vision_uncertainty=vision_uncertainty,
            evidence_count=evidence_count,
        )


# Backward Compatibility Alias
CompositeConfidenceCalculator = ConfidenceCalculator
