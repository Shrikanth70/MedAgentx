"""Agent 1: Query Planner & Reformulator Module."""

import re
from typing import Any
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger()


class PlannerOutput(BaseModel):
    """Output contract for Agent 1 (Planner)."""

    original_query: str
    rephrased_query: str = ""
    decomposed_subqueries: list[str] = Field(default_factory=list)
    extracted_entities: list[str] = Field(default_factory=list)
    retrieval_strategy: str = "hybrid"


class PlannerAgent:
    """Agent 1: Query Planner & Reformulator with Dynamic Adaptation."""

    def __init__(self):
        self.name = "Agent 1 - Planner & Reformulator"

    def plan(self, query_text: str) -> PlannerOutput:
        """Decompose clinical query and extract entities."""
        logger.info("Executing Agent 1 (Planner)", query=query_text)

        # Extract medical terms/entities via regex & heuristics
        words = re.findall(r"\b[A-Z][a-zA-Z0-9\-]+\b|\b(?:STEMI|NSTEMI|ACS|DAPT|HFrEF|SGLT2|CAP|eGFR|AKI|COPD|DKA|Sepsis)\b", query_text, re.IGNORECASE)
        extracted_entities = list(set(words)) if words else ["General Clinical"]

        # Decompose subqueries
        subqueries = [
            query_text,
            f"Clinical practice guidelines for {query_text}",
            f"Pharmacology and contraindications for {query_text}",
        ]

        return PlannerOutput(
            original_query=query_text,
            rephrased_query=query_text,
            decomposed_subqueries=subqueries,
            extracted_entities=extracted_entities,
            retrieval_strategy="hybrid",
        )

    def plan_patient(self, patient: Any) -> PlannerOutput:
        """Decompose clinical presentation of a real patient and formulate targeted queries."""
        summary = patient.summarize_clinical_state()
        logger.info("Executing Agent 1 for Real Patient Case", patient_id=getattr(patient, "patient_id", "PT-001"))

        # Extract focal diseases and conditions
        entities = list(getattr(patient, "past_medical_history", []))
        if getattr(patient, "chief_complaint", ""):
            entities.append(patient.chief_complaint)
        if getattr(patient, "is_hypotensive", False):
            entities.append("Hypotension")
        if getattr(patient, "is_severe_renal_impairment", False):
            entities.append("Acute Kidney Injury")
        if getattr(patient, "is_hyperkalemic", False):
            entities.append("Hyperkalemia")

        search_query = f"{patient.chief_complaint} {' '.join(patient.past_medical_history)} {patient.clinical_question}".strip()

        subqueries = [
            search_query,
            f"Guideline management for {patient.chief_complaint}",
            f"Contraindications in {patient.chief_complaint} with eGFR {patient.egfr or 'normal'} and BP {patient.sbp or 'normal'}",
        ]

        return PlannerOutput(
            original_query=search_query,
            rephrased_query=summary,
            decomposed_subqueries=subqueries,
            extracted_entities=list(set(entities)) if entities else ["Clinical Evaluation"],
            retrieval_strategy="hybrid",
        )

    def adapt_query(
        self,
        original_query: str,
        retrieval_gap: str,
        missing_evidence: list[str],
        previous_queries: list[str],
    ) -> str:
        """Generate targeted search query dynamically addressing identified retrieval gap."""
        q_lower = original_query.lower()

        if missing_evidence:
            gap_focus = " ".join(missing_evidence)
            adapted = f"{original_query} {gap_focus}"
        elif retrieval_gap == "INSUFFICIENT_RELEVANCE":
            adapted = f"Clinical practice guidelines for {original_query}"
        elif retrieval_gap == "MISSING_EVIDENCE" and ("stemi" in q_lower or "dapt" in q_lower):
            adapted = f"{original_query} reperfusion strategy primary PCI fibrinolysis P2Y12 selection ticagrelor prasugrel clopidogrel"
        elif retrieval_gap == "MISSING_EVIDENCE" and ("hfref" in q_lower or "heart failure" in q_lower):
            adapted = f"{original_query} GDMT four pillars ARNI beta blocker MRA SGLT2 inhibitor guidelines"
        else:
            adapted = f"Evidence guidelines protocol for {original_query}"

        # Avoid exact duplicate of previous query
        if adapted in previous_queries:
            adapted = f"{adapted} detailed management guideline"

        logger.info("Adapted retrieval query for next iteration", original=original_query, adapted=adapted, gap=retrieval_gap)
        return adapted
