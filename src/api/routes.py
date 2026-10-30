"""FastAPI Routes and Endpoints for MedAgentX."""

import functools
from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel
from config.settings import settings
from src.knowledge.vector_store import VectorStoreClient
from src.knowledge.graph_store import GraphStoreClient
from src.orchestrator.system import MedAgentXSystem

router = APIRouter()


@functools.lru_cache(maxsize=1)
def _get_system() -> MedAgentXSystem:
    """Lazy singleton: only instantiated on first API call, not at import time."""
    return MedAgentXSystem()


class ConsultRequest(BaseModel):
    """Clinical consultation query payload."""

    query: str
    system_id: str = "medagentx"  # "medagentx", "b0", "b1", "b2", "b7"


class PatientConsultRequest(BaseModel):
    """Patient clinical vignette or structured case consultation payload."""

    vignette_text: str = ""
    patient_data: dict[str, Any] | None = None


@router.get("/health", tags=["Health"])
async def root_health():
    """Root health check endpoint."""
    return {"status": "ok", "app": settings.APP_NAME, "env": settings.APP_ENV}


@router.get("/api/v1/health", tags=["Health"])
async def api_health():
    """Detailed API health probe checking database connections."""
    vector_client = VectorStoreClient()
    graph_client = GraphStoreClient()

    vector_status = vector_client.health_check()
    graph_status = graph_client.health_check()

    overall_healthy = (
        vector_status.get("status") == "healthy"
        and graph_status.get("status") == "healthy"
    )

    return {
        "status": "healthy" if overall_healthy else "degraded",
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "services": {
            "qdrant": vector_status,
            "neo4j": graph_status,
        },
    }


@router.post("/api/v1/consult", tags=["Clinical CDSS"])
async def consult_clinical_decision(payload: ConsultRequest) -> dict[str, Any]:
    """Execute clinical decision support consultation through ACMR pipeline."""
    res = _get_system().query(payload.query)

    return {
        "system_id": res.system_id,
        "query": res.query,
        "answer": res.answer,
        "evidence_chunks": res.evidence_chunks,
        "citations": res.citations,
        "confidence": res.confidence,
        "latency_seconds": res.latency_seconds,
        "token_count": res.token_count,
        "metrics": {
            "s_vec": res.metadata.get("s_vec", 0.0),
            "s_graph": res.metadata.get("s_graph", 0.0),
            "v_agree": res.metadata.get("v_agree", 0.0),
            "r_contra": res.metadata.get("r_contra", 0.0),
            "active_agent_count": res.metadata.get("active_agent_count", 1),
            "retrieval_rounds": res.metadata.get("retrieval_rounds", 1),
            "route_decision": res.metadata.get("route_decision", "N/A"),
        },
        "trace": res.metadata.get("trace", []),
    }


@router.post("/api/v1/patient-consult", tags=["Clinical CDSS"])
async def patient_consult_decision(payload: PatientConsultRequest) -> dict[str, Any]:
    """Execute real-world patient decision support consultation with vital & lab safety guards."""
    from src.orchestrator.patient import PatientCase, PatientVignetteParser

    if payload.patient_data:
        patient = PatientCase(**payload.patient_data)
    elif payload.vignette_text:
        patient = PatientVignetteParser.parse(payload.vignette_text)
    else:
        patient = PatientCase(chief_complaint="General Clinical Consultation")

    res = _get_system().query_patient(patient)

    return {
        "system_id": res.system_id,
        "patient_id": getattr(patient, "patient_id", "PT-001"),
        "chief_complaint": getattr(patient, "chief_complaint", ""),
        "report": res.answer,
        "confidence": res.confidence,
        "route_decision": res.metadata.get("route_decision", "N/A"),
        "safety_severity": res.metadata.get("safety_severity", "NONE"),
        "contradiction_type": res.metadata.get("contradiction_type", "NONE"),
        "latency_seconds": res.latency_seconds,
        "token_count": res.token_count,
        "citations": res.citations,
        "trace": res.metadata.get("trace", []),
    }
