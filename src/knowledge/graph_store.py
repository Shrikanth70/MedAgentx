"""Neo4j Knowledge Graph Interface for MedAgentX."""

from typing import Any
import structlog
from neo4j import GraphDatabase
from config.settings import settings

logger = structlog.get_logger()


class GraphStoreClient:
    """Wrapper client for Neo4j Knowledge Graph Database."""

    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None

    def connect(self):
        """Lazy driver connection."""
        if not self._driver:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self):
        """Close driver session."""
        if self._driver:
            self._driver.close()

    def health_check(self) -> dict[str, Any]:
        """Verify Neo4j database health."""
        try:
            driver = self.connect()
            with driver.session() as session:
                result = session.run("RETURN 1 AS health")
                record = result.single()
                if record and record["health"] == 1:
                    return {"status": "healthy", "uri": self.uri}
            return {"status": "unhealthy", "uri": self.uri, "error": "Invalid health query return"}
        except Exception as e:
            logger.error("Neo4j health check failed", error=str(e))
            return {"status": "unhealthy", "uri": self.uri, "error": str(e)}

    def execute_cypher(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Execute arbitrary Cypher query."""
        try:
            driver = self.connect()
            with driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error("Cypher execution failed", query=query, error=str(e))
            return []
