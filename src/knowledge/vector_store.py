"""Qdrant Vector Database Interface for MedAgentX."""

from typing import Any
import structlog
from qdrant_client import QdrantClient
from qdrant_client.http import models
from config.settings import settings

logger = structlog.get_logger()


class VectorStoreClient:
    """Wrapper client for Qdrant Vector Database."""

    def __init__(self, url: str = settings.QDRANT_URL):
        self.url = url
        self._client: QdrantClient | None = None
        self.collection_name = settings.QDRANT_COLLECTION

    @property
    def client(self) -> QdrantClient:
        """Lazy connection: only connect when first accessed."""
        if self._client is None:
            self._client = QdrantClient(url=self.url)
        return self._client

    def health_check(self) -> dict[str, Any]:
        """Verify Qdrant server health."""
        try:
            collections = self.client.get_collections()
            return {
                "status": "healthy",
                "url": self.url,
                "collections_count": len(collections.collections),
            }
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return {"status": "unhealthy", "url": self.url, "error": str(e)}

    def initialize_collection(self, vector_size: int = 768) -> bool:
        """Create Qdrant collection if not already existing."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
                logger.info("Created Qdrant collection", collection=self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to initialize Qdrant collection", error=str(e))
            return False

    def search(self, query_vector: list[float], top_k: int = settings.VECTOR_TOP_K) -> list[dict[str, Any]]:
        """Search top-k nearest vectors."""
        try:
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                )
            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload,
                }
                for hit in results
            ]
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []
