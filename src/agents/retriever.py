"""Agent 2: Bi-Modal Hybrid Retriever Module."""

from typing import Any
from pydantic import BaseModel, Field
import structlog
from config.settings import settings
from src.knowledge.hybrid_retriever import HybridRetriever

logger = structlog.get_logger()


class RetrieverOutput(BaseModel):
    """Output contract for Agent 2 (Retriever)."""

    s_vec: float = 0.0
    s_graph: float = 0.0
    vector_evidence: list[dict[str, Any]] = Field(default_factory=list)
    graph_evidence: list[dict[str, Any]] = Field(default_factory=list)


class RetrieverAgent:
    """Agent 2: Bi-Modal Hybrid Retriever."""

    def __init__(self):
        self.name = "Agent 2 - Hybrid Retriever"
        self.hybrid_retriever = HybridRetriever()

    def retrieve(
        self, query_text: str, entities: list[str] | None = None, top_k: int = settings.VECTOR_TOP_K
    ) -> RetrieverOutput:
        """Execute hybrid vector search and knowledge graph traversal."""
        logger.info("Executing Agent 2 (Retriever)", query=query_text, entities=entities)
        results = self.hybrid_retriever.retrieve(query=query_text, top_k=top_k)
        vector_docs = results.get("vector_evidence", [])
        graph_triples = results.get("graph_evidence", [])

        s_vec = 0.85 if vector_docs else 0.0
        s_graph = 0.90 if graph_triples else 0.0

        return RetrieverOutput(
            s_vec=s_vec,
            s_graph=s_graph,
            vector_evidence=vector_docs,
            graph_evidence=graph_triples,
        )

    def retrieve_evidence(
        self, query_text: str, entities: list[str] | None = None, top_k: int = settings.VECTOR_TOP_K
    ) -> RetrieverOutput:
        """Alias for retrieve method."""
        return self.retrieve(query_text=query_text, entities=entities, top_k=top_k)
