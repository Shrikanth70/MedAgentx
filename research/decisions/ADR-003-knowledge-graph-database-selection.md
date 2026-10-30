# ADR-003: Knowledge Graph Database Selection (Neo4j)

## Context
MedAgentX relies on structured medical knowledge representations (UMLS / MeSH / SNOMED CT) to execute multi-hop path traversal and prevent false semantic associations in vector search.

## Decision
Select **Neo4j Community Edition (Docker Container `v5.17.0`)** with the APOC library plugin enabled.

## Alternatives Considered
1. **In-Memory NetworkX:** Simple Python graph library, but lacks persistent Cypher query indexing and memory scalability for multi-thousand node medical ontologies.
2. **Memgraph:** Fast C++ graph database, but less widespread community Cypher tool support in LangChain/LangGraph ecosystems.
3. **Neo4j:** Standard graph database with native Cypher query engine, APOC procedure extensions, robust Python drivers, and containerized deployment.

## Status
**ACCEPTED** (Date: August 13, 2026)
