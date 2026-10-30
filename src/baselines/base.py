"""Abstract Base Interface for Baselines and MedAgentX Systems."""

from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel, Field


class SystemResponse(BaseModel):
    """Unified Response Contract across all systems and baselines."""

    system_id: str
    query: str
    answer: str
    evidence_chunks: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    latency_seconds: float = 0.0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseSystem(ABC):
    """Abstract Base Class for Systems."""

    def __init__(self, system_id: str):
        self.system_id = system_id

    @abstractmethod
    def query(self, user_query: str) -> SystemResponse:
        """Execute clinical query through system."""
        pass
