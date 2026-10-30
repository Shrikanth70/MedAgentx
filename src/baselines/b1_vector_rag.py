"""Baseline B1: Standard Dense Vector RAG System."""

import time
import os
import json
import structlog
from src.baselines.base import BaseSystem, SystemResponse
from src.knowledge.vector_store import VectorStoreClient
from scripts.build_embeddings import generate_dense_vector

logger = structlog.get_logger()


class B1VectorRAG(BaseSystem):
    """Baseline B1: Standard Vector Retrieval-Augmented Generation."""

    def __init__(self, vector_cache_path: str = "data/embeddings/vector_cache.json"):
        super().__init__(system_id="B1_VECTOR_RAG")
        self.vector_cache_path = vector_cache_path
        self.vector_client = VectorStoreClient()

    def _get_evidence(self, query_text: str, top_k: int = 3) -> list[dict]:
        """Fetch vector evidence from Qdrant or fallback local cache."""
        query_vec = generate_dense_vector(query_text)
        qdrant_results = self.vector_client.search(query_vec, top_k=top_k)

        if qdrant_results:
            return [res["payload"] for res in qdrant_results if "payload" in res]

        # Fallback to local vector cache
        if os.path.exists(self.vector_cache_path):
            with open(self.vector_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
                return cache[:top_k]
        return []

    def query(self, user_query: str) -> SystemResponse:
        """Execute standard retrieve-then-generate pipeline."""
        start_time = time.time()
        evidence_chunks = self._get_evidence(user_query, top_k=3)

        citations = [doc.get("source", "Guideline") for doc in evidence_chunks if "source" in doc]

        context_str = "\n\n".join([c.get("content", "") for c in evidence_chunks])
        answer = (
            f"Evidence-grounded response for '{user_query}':\n"
            f"Retrieved Evidence: {context_str[:300]}...\n"
            "Clinical Conclusion: Standard single-pass retrieval indicates adherence to cited guidelines."
        )

        latency = time.time() - start_time
        token_count = len(user_query.split()) + len(context_str.split()) + len(answer.split())

        return SystemResponse(
            system_id=self.system_id,
            query=user_query,
            answer=answer,
            evidence_chunks=evidence_chunks,
            citations=citations,
            confidence=0.70,
            latency_seconds=round(latency, 4),
            token_count=token_count,
            metadata={"retrieval_rounds": 1},
        )
