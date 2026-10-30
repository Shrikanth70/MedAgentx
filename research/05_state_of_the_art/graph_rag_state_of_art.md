# State of the Art: Graph RAG & Knowledge Graph Integration

## 1. Executive Summary
Graph RAG combines unstructured text vector search with structured Knowledge Graph (KG) representations to solve relational and multi-hop query limitations inherent to dense vector search alone.

## 2. Technical Comparison

| Feature | Vector RAG (Qdrant / FAISS) | Knowledge Graph RAG (Neo4j / GraphRAG) | Hybrid Vector + KG (MedAgentX) |
| :--- | :--- | :--- | :--- |
| **Search Mechanism** | Top-k cosine similarity | Entity-relation graph traversal | Bi-modal dense matching + multi-hop graph expansion |
| **Multi-Hop Performance** | Poor (semantic drift) | Excellent (explicit paths) | State-of-the-Art (semantic context + path validation) |
| **Global Sensemaking** | Fails on corpus-wide trends | Strong (community summaries) | Strong |
| **Factual Grounding** | Medium (embedding hallucination) | High (ontology verified) | High (ontology verified) |
| **Indexing Overhead** | Low / Fast | High / Compute-intensive | Selective entity graph extraction |
