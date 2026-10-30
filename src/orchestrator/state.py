"""MedAgentX Workflow Execution State Schema."""

from typing import Any
from pydantic import BaseModel, Field


class MedAgentXState(BaseModel):
    """Pydantic State Schema maintained throughout ACMR execution."""

    query: str
    iteration: int = 0
    max_iterations: int = 2
    c_total: float = 0.0

    s_vec: float = 0.0
    s_graph: float = 0.0
    v_agree: float = 0.0
    r_contra: float = 0.0

    planner_data: dict[str, Any] = Field(default_factory=dict)
    vector_evidence: list[dict[str, Any]] = Field(default_factory=list)
    graph_evidence: list[dict[str, Any]] = Field(default_factory=list)
    verified_evidence: list[dict[str, Any]] = Field(default_factory=list)
    detected_contradictions: list[str] = Field(default_factory=list)

    active_agent_count: int = 0
    token_count: int = 0
    route_decision: str = "INIT"  # "STOP", "RE_RETRIEVE", "ABSTAIN"
    status: str = "RUNNING"       # "COMPLETED", "ABSTAINED", "FAILED"
    report: str = ""
    citations: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
