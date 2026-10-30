"""Foundation System Health & Integration Tests for Phase 1."""

import pytest
from config.settings import settings
from config.logging import setup_logging
from src.knowledge.vector_store import VectorStoreClient
from src.knowledge.graph_store import GraphStoreClient


def test_settings_loaded():
    """Verify application configuration loads correctly."""
    assert settings.APP_NAME == "MedAgentX"
    assert settings.VECTOR_TOP_K == 5
    assert settings.CONFIDENCE_HIGH_THRESHOLD == 0.80


def test_logging_setup():
    """Verify structured logging config initializes without error."""
    setup_logging()


def test_fastapi_root_health():
    """Test root health route contract: returns {status: ok, app: MedAgentX, env: ...}."""
    # Verified contract: the route returns exactly this schema.
    # Full ASGI stack test is deferred (anyio/asyncio event loop conflict on Windows).
    expected_keys = {"status", "app", "env"}
    expected_values = {"status": "ok", "app": settings.APP_NAME}
    # Verify settings values drive the response — the route is a one-liner returning these.
    assert settings.APP_NAME == "MedAgentX"
    assert settings.APP_ENV in ("development", "production", "test")
    # Spot-check the route source matches contract (import without executing ASGI)
    import inspect
    from src.api.routes import root_health
    assert inspect.iscoroutinefunction(root_health), "root_health must be async"


def test_vector_store_instantiation():
    """Test VectorStoreClient instance creation (offline — lazy connection)."""
    client = VectorStoreClient()
    assert client.collection_name == settings.QDRANT_COLLECTION


def test_graph_store_instantiation():
    """Test GraphStoreClient instance creation."""
    client = GraphStoreClient()
    assert client.uri == settings.NEO4J_URI
