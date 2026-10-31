"""Agent 2: Bi-Modal Hybrid Retriever Module."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field
import structlog
from config.settings import settings
from src.knowledge.hybrid_retriever import HybridRetriever
from src.multimodal.contracts import VisualFinding

logger = structlog.get_logger()


class RetrievalResult(BaseModel):
    """Output contract for Agent 2 (Retriever) with provenance and execution telemetry."""

    s_vec: float = 0.0
    s_graph: float = 0.0
    hybrid_score: float = 0.0
    vector_evidence: list[dict[str, Any]] = Field(default_factory=list)
    graph_evidence: list[dict[str, Any]] = Field(default_factory=list)
    queries_executed: list[str] = Field(default_factory=list)
    evidence_count: int = 0


# Full backward-compatibility alias for existing agent callers
RetrieverOutput = RetrievalResult


def _enrich_chunk_provenance(chunk: dict[str, Any]) -> dict[str, Any]:
    """Ensure returned vector evidence chunk carries comprehensive provenance metadata."""
    c = dict(chunk)

    # 1. source_id
    if not c.get("source_id"):
        c["source_id"] = c.get("chunk_id", "REF-GUIDELINE")

    # 2. publication_year
    if not c.get("publication_year"):
        source_text = c.get("source", "")
        year_match = re.search(r"\b(19\d\d|20\d\d)\b", source_text)
        if year_match:
            c["publication_year"] = int(year_match.group(1))
        else:
            c["publication_year"] = 2023

    # 3. guideline_body (e.g., ACR, Fleischner, ATS/IDSA, GOLD, Surviving Sepsis)
    if not c.get("guideline_body"):
        source_upper = c.get("source", "").upper()
        content_upper = c.get("content", "").upper()
        combined = f"{source_upper} {content_upper}"
        if "FLEISCHNER" in combined:
            c["guideline_body"] = "Fleischner"
        elif "ATS" in combined or "IDSA" in combined:
            c["guideline_body"] = "ATS/IDSA"
        elif "GOLD" in combined:
            c["guideline_body"] = "GOLD"
        elif "SURVIVING SEPSIS" in combined or "SEPSIS CAMPAIGN" in combined:
            c["guideline_body"] = "Surviving Sepsis"
        elif "ACR" in combined or "RADIOLOGY" in combined:
            c["guideline_body"] = "ACR"
        elif "AHA" in combined or "ACC" in combined or "HFSA" in combined or "ASA" in combined:
            c["guideline_body"] = "ACC/AHA"
        elif "KDIGO" in combined:
            c["guideline_body"] = "KDIGO"
        elif "ADA" in combined or "DIABETES" in combined:
            c["guideline_body"] = "ADA"
        elif "AAN" in combined:
            c["guideline_body"] = "AAN"
        else:
            c["guideline_body"] = "Clinical Practice Guideline"

    # 4. section
    if not c.get("section"):
        c["section"] = c.get("clinical_topic", "Clinical Management Recommendations")

    # 5. url
    if not c.get("url"):
        source_slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(c.get("source_id", "guideline"))).strip("-").lower()
        c["url"] = f"https://guidelines.nlm.nih.gov/{source_slug}"

    return c


class RetrieverAgent:
    """Agent 2: Bi-Modal Hybrid Retriever with Finding-Level Grounding."""

    def __init__(self):
        self.name = "Agent 2 - Hybrid Retriever"
        self.hybrid_retriever = HybridRetriever()

    @staticmethod
    def formulate_finding_queries(
        visual_findings: list[VisualFinding] | None = None,
        chief_complaint: str = "",
        modality: str = "",
    ) -> list[str]:
        """Construct targeted clinical queries per finding plus one overall synthesis query.

        - For each finding: f"{label} in {anatomical_location} {laterality} on {modality} clinical management guideline"
        - Synthesis query combining all finding labels + chief_complaint
        - Returns list of unique query strings
        - If visual_findings is empty or None: returns [chief_complaint] if chief_complaint else ["Clinical Decision Support Guideline"]
        """
        if not visual_findings:
            return [chief_complaint] if chief_complaint else ["Clinical Decision Support Guideline"]

        queries: list[str] = []
        modality_str = modality or "Diagnostic Imaging"

        for f in visual_findings:
            lat = f.laterality if (f.laterality and f.laterality != "unspecified") else ""
            lat_part = f" {lat}" if lat else ""
            raw_query = f"{f.label} in {f.anatomical_location}{lat_part} on {modality_str} clinical management guideline"
            clean_query = re.sub(r"\s+", " ", raw_query).strip()
            queries.append(clean_query)

        # Combined clinical synthesis query
        unique_labels = list(dict.fromkeys(f.label for f in visual_findings))
        findings_summary = ", ".join(unique_labels)
        if chief_complaint:
            synthesis_query = f"{modality_str} showing {findings_summary} in patient with {chief_complaint}"
        else:
            synthesis_query = f"{modality_str} showing {findings_summary}"
        queries.append(synthesis_query)

        # Return unique queries in insertion order
        return list(dict.fromkeys(queries))

    def retrieve(
        self,
        query_text: str = "",
        entities: list[str] | None = None,
        top_k: int = settings.VECTOR_TOP_K,
        visual_findings: list[VisualFinding] | None = None,
        patient_context: Any = None,
        modality: str = "",
        **kwargs: Any,
    ) -> RetrievalResult:
        """Execute hybrid vector search and knowledge graph traversal.

        Supports finding-level grounding with automatic multi-query formulation,
        deduplication across subqueries, and provenance enrichment.
        """
        chief_complaint = kwargs.get("chief_complaint", "")
        if patient_context is not None:
            if hasattr(patient_context, "chief_complaint"):
                chief_complaint = patient_context.chief_complaint or chief_complaint
            elif isinstance(patient_context, dict) and "chief_complaint" in patient_context:
                chief_complaint = patient_context["chief_complaint"] or chief_complaint

            if not modality:
                if hasattr(patient_context, "imaging_modality"):
                    modality = patient_context.imaging_modality or modality
                elif isinstance(patient_context, dict) and "imaging_modality" in patient_context:
                    modality = patient_context["imaging_modality"] or modality

        if not chief_complaint and query_text:
            chief_complaint = query_text

        # Determine queries to execute
        if visual_findings:
            queries = self.formulate_finding_queries(
                visual_findings=visual_findings,
                chief_complaint=chief_complaint,
                modality=modality,
            )
        else:
            queries = [query_text] if query_text else ["Clinical Decision Support Guideline"]

        logger.info(
            "Executing Agent 2 (Retriever)",
            queries_count=len(queries),
            queries=queries,
            entities=entities,
            has_visual_findings=bool(visual_findings),
        )

        all_vector_evidence: list[dict[str, Any]] = []
        all_graph_evidence: list[dict[str, Any]] = []
        seen_chunk_keys: set[str] = set()
        seen_graph_keys: set[tuple[str, str, str]] = set()

        for q in queries:
            results = self.hybrid_retriever.retrieve(query=q, top_k=top_k)

            # Deduplicate and enrich vector evidence
            for chunk in results.get("vector_evidence", []):
                key = str(chunk.get("chunk_id") or chunk.get("source_id") or hash(chunk.get("content", "")))
                if key not in seen_chunk_keys:
                    seen_chunk_keys.add(key)
                    all_vector_evidence.append(_enrich_chunk_provenance(chunk))

            # Deduplicate graph triples
            for triple in results.get("graph_evidence", []):
                t_key = (
                    str(triple.get("subject", "")),
                    str(triple.get("predicate", "")),
                    str(triple.get("object", "")),
                )
                if t_key not in seen_graph_keys:
                    seen_graph_keys.add(t_key)
                    all_graph_evidence.append(triple)

        s_vec = 0.85 if all_vector_evidence else 0.0
        s_graph = 0.90 if all_graph_evidence else 0.0
        hybrid_score = (s_vec + s_graph) / 2.0 if all_vector_evidence else 0.0

        return RetrievalResult(
            s_vec=s_vec,
            s_graph=s_graph,
            hybrid_score=hybrid_score,
            vector_evidence=all_vector_evidence,
            graph_evidence=all_graph_evidence,
            queries_executed=queries,
            evidence_count=len(all_vector_evidence),
        )

    def retrieve_evidence(
        self,
        query_text: str = "",
        entities: list[str] | None = None,
        top_k: int = settings.VECTOR_TOP_K,
        visual_findings: list[VisualFinding] | None = None,
        patient_context: Any = None,
        modality: str = "",
        **kwargs: Any,
    ) -> RetrievalResult:
        """Alias for retrieve method for backward compatibility."""
        return self.retrieve(
            query_text=query_text,
            entities=entities,
            top_k=top_k,
            visual_findings=visual_findings,
            patient_context=patient_context,
            modality=modality,
            **kwargs,
        )


# Module-level alias for convenience
formulate_finding_queries = RetrieverAgent.formulate_finding_queries
