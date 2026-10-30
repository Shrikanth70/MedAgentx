# ADR-002: Vector Database Selection (Qdrant)

## Context
MedAgentX requires a high-performance vector database to store and retrieve dense embeddings generated from PubMed abstracts and clinical guidelines.

## Decision
Select **Qdrant (Docker Container `v1.8.0`)** as the primary vector store for MedAgentX.

## Alternatives Considered
1. **ChromaDB:** Lightweight embedded option, but limited in production HNSW filtering and payload indexing at scale.
2. **Milvus:** Powerful distributed vector DB, but heavy operational footprint for single-machine research experiments.
3. **Qdrant:** Native Rust engine, low memory footprint, rich payload metadata filtering, official Python gRPC/REST client.

## Advantages
- Excellent HNSW index recall and query throughput.
- Full payload filtering support (e.g., filter by source type `guideline`, publication year, or MeSH terms).
- Containerized deployment via Docker Compose with volume persistence.

## Status
**ACCEPTED** (Date: August 13, 2026)
