"""Composite Evidence Confidence Calculator ($C_{total}$) for MedAgentX."""

import structlog

logger = structlog.get_logger()


class ConfidenceCalculator:
    """Calculates Composite Evidence Confidence Score ($C_{total}$) with Relevance & Completeness Weighting."""

    @staticmethod
    def calculate_composite_confidence(
        s_vec: float,
        s_graph: float,
        v_agree: float,
        r_contra: float,
        s_rel: float = 1.0,
        completeness: float = 1.0,
        novelty: float = 1.0,
        contradiction_type: str = "NONE",
    ) -> float:
        """Calculate composite confidence score ($C_{total}$) gated by Evidence Relevance ($S_{rel}$) & Completeness.
        
        Formula:
            If CLINICAL_CONTRAINDICATION: Confidence measures strong evidence grounding for safety constraint (r_contra not subtracted from C_total).
            Else: C_total = 0.30*(S_vec*S_rel) + 0.20*(S_graph*S_rel) + 0.20*V_agree + 0.20*completeness + 0.10*novelty - 0.15*R_contra
        """
        effective_s_vec = s_vec * s_rel
        effective_s_graph = s_graph * s_rel

        contra_penalty = 0.15 * r_contra if contradiction_type != "CLINICAL_CONTRAINDICATION" else 0.0

        raw_score = (
            0.30 * effective_s_vec
            + 0.20 * effective_s_graph
            + 0.20 * v_agree
            + 0.20 * completeness
            + 0.10 * novelty
            - contra_penalty
        )
        c_total = max(0.0, min(1.0, raw_score))
        logger.info(
            "Calculated Composite Confidence",
            c_total=c_total,
            s_vec=s_vec,
            s_graph=s_graph,
            v_agree=v_agree,
            r_contra=r_contra,
            s_rel=s_rel,
            completeness=completeness,
            novelty=novelty,
            contradiction_type=contradiction_type,
        )
        return round(c_total, 4)

    @classmethod
    def calculate(
        cls,
        s_vec: float,
        s_graph: float,
        v_agree: float,
        r_contra: float,
        s_rel: float = 1.0,
        completeness: float = 1.0,
        novelty: float = 1.0,
        contradiction_type: str = "NONE",
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
        )


# Backward Compatibility Alias
CompositeConfidenceCalculator = ConfidenceCalculator
